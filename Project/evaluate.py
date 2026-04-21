'''
Evaluate a trained GPT + tokenizer pair on a held-out text file.

Reports:
  - cross-entropy loss (nats)
  - perplexity (per token under this tokenizer)   <-- NOT comparable across tokenizers
  - bits per character (BPC)                      <-- comparable across tokenizers

BPC derivation:
  loss_in_nats = average per-token negative log-likelihood
  loss_in_bits = loss_in_nats * log2(e)
  total_bits   = loss_in_bits * num_tokens
  bpc          = total_bits / num_chars_in_text

BPC normalizes the model's surprisal to the raw text, so a model using a
more efficient tokenizer (fewer tokens per char) does not get "free"
perplexity wins.

Usage:
  python evaluate.py \
      --weights artifacts/weights/medical/model_weights.pt \
      --tokenizer artifacts/tokenizers/medical \
      --text artifacts/data/pubmed_eval.txt
'''

import argparse
import math
import os
import torch

from gpt import GPTModel
from tokenizer import HFTokenizer
from train import pick_device


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

    with open(text_path, "r", encoding="utf-8") as f:
        samples = [line.rstrip("\n") for line in f if line.strip()]

    total_chars = sum(len(s) for s in samples)

    # Tokenize and pack the eval text the same way we do for training, then
    # compute average CE loss over all predicted positions.
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
        logits = model(x)                           # (B, S, V)
        # per-element CE, summed
        loss = loss_fn(logits.transpose(1, 2), y)   # scalar, summed over all (B,S)
        total_nll += loss.item()
        total_predicted_tokens += y.numel()

    avg_nll_nats = total_nll / total_predicted_tokens
    ppl = math.exp(avg_nll_nats)
    total_bits = avg_nll_nats * math.log2(math.e) * total_predicted_tokens
    bpc = total_bits / max(total_chars, 1)

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


if __name__ == "__main__":
    main()
