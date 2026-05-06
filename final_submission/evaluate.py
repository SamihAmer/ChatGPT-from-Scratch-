"""
evaluate.py - score a trained GPT against a held-out text file.

Reports per-token cross-entropy, perplexity, and bits-per-character (BPC).

BPC is the metric this project relies on, since perplexity is per-token and
therefore not comparable across tokenizers (a tokenizer that compresses more
makes each token cover more text and thus harder to predict). BPC normalizes
to characters, so two models with different tokenizers can be compared fairly:

    BPC = log2(e) * avg_nll_nats * n_tokens / n_chars

Example:
    python evaluate.py --weights artifacts/weights/medical/model_weights.pt \\
                       --tokenizer artifacts/tokenizers/medical \\
                       --text artifacts/data/pubmed_eval.txt
"""

import argparse
import csv
import math
import os
import torch

from gpt import GPTModel
from tokenizer import HFTokenizer
from train import pick_device


# Columns written to artifacts/results/eval_table.csv when --csv-append is set.
EVAL_CSV_COLUMNS = [
    "seed", "tokenizer_name", "checkpoint_step", "eval_corpus",
    "bpc", "ppl", "nll_nats", "n_tokens", "n_chars",
    "weights_path",
]


def _append_eval_row(csv_path, row):
    # Header is written automatically the first time we write to a fresh file.
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_CSV_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in EVAL_CSV_COLUMNS})


@torch.no_grad()
def evaluate(weights_path, tokenizer_dir, text_path,
             d_model=512, n_heads=8, layers=6, seq_len=256, batch_size=16):
    device = pick_device()

    tok = HFTokenizer(tokenizer_dir)
    tok.load()
    eos_id = tok.eos_token_id
    vocab_size = tok.vocab_size

    model = GPTModel(
        d_model=d_model, n_heads=n_heads, layers=layers,
        vocab_size=vocab_size, max_seq_len=seq_len,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Read eval text, encode the same way as training (one doc per line, EOS
    # between docs, then chunked into seq_len+1 windows).
    with open(text_path, "r", encoding="utf-8") as f:
        samples = [line.rstrip("\n") for line in f if line.strip()]

    total_chars = sum(len(s) for s in samples)

    all_tokens = []
    for s in samples:
        ids = tok.encode(s)
        ids.append(eos_id)
        all_tokens.extend(ids)

    chunk = seq_len + 1
    n_seq = len(all_tokens) // chunk
    if n_seq == 0:
        raise RuntimeError(f"Eval corpus too small: {len(all_tokens)} tokens, need >= {chunk}")
    all_tokens = all_tokens[: n_seq * chunk]
    arr = torch.tensor(all_tokens, dtype=torch.long).view(n_seq, chunk)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    total_nll = 0.0
    total_predicted_tokens = 0
    for i in range(0, n_seq, batch_size):
        batch = arr[i:i + batch_size].to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = model(x)
        loss = loss_fn(logits.transpose(1, 2), y)
        total_nll += loss.item()
        total_predicted_tokens += y.numel()

    avg_nll_nats = total_nll / total_predicted_tokens
    ppl = math.exp(avg_nll_nats)
    bpc = (avg_nll_nats * math.log2(math.e) * total_predicted_tokens) / max(total_chars, 1)

    return {
        "text": text_path,
        "tokenizer": tokenizer_dir,
        "weights": weights_path,
        "num_chars": total_chars,
        "num_tokens_eval": total_predicted_tokens,
        "tokens_per_char": total_predicted_tokens / max(total_chars, 1),
        "avg_nll_nats": avg_nll_nats,
        "perplexity": ppl,
        "bpc": bpc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    # Sweep metadata: when --csv-append is set, one row gets appended to
    # the named CSV with the four labels below filled in.
    ap.add_argument("--csv-append", default="",
                    help="If set, append a result row to this CSV.")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--tokenizer-name", default="")
    ap.add_argument("--checkpoint-step", type=int, default=-1)
    ap.add_argument("--eval-corpus", default="")
    args = ap.parse_args()

    r = evaluate(args.weights, args.tokenizer, args.text,
                 d_model=args.d_model, n_heads=args.n_heads, layers=args.layers,
                 seq_len=args.seq_len, batch_size=args.batch_size)

    print("\n=== eval result ===")
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:>22}: {v:.4f}")
        else:
            print(f"  {k:>22}: {v}")

    if args.csv_append:
        row = {
            "seed": args.seed,
            "tokenizer_name": args.tokenizer_name,
            "checkpoint_step": args.checkpoint_step,
            "eval_corpus": args.eval_corpus,
            "bpc": r["bpc"],
            "ppl": r["perplexity"],
            "nll_nats": r["avg_nll_nats"],
            "n_tokens": r["num_tokens_eval"],
            "n_chars": r["num_chars"],
            "weights_path": args.weights,
        }
        _append_eval_row(args.csv_append, row)
        print(f"  [csv] appended row to {args.csv_append}")


if __name__ == "__main__":
    main()
