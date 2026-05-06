"""
train.py - GPT training loop, parameterized so the same script trains
every tokenizer variant in this experiment.

Logs (step, tokens_seen, chars_seen, loss) to a CSV during training.
The chars_seen column is the one that lets us compare models trained
under different tokenizers fairly: a tokenizer that compresses more
hits the same step count with more raw text exposure.

Example:
    python train.py --dataset artifacts/datasets/pubmed_train_medical.npy \\
                    --tokenizer artifacts/tokenizers/medical \\
                    --out artifacts/weights/medical \\
                    --epochs 2 --seed 1
"""

import argparse
import csv
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from gpt import GPTModel
from tokenizer import HFTokenizer


def _seed_everything(seed):
    # Same seed -> same model init, same DataLoader shuffle. CuDNN deterministic
    # mode is left off (it's a 10-20% throughput hit and we don't need bit-exact
    # replays, just controlled run-to-run variation).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            prog = float(stepnum) / float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            steps_after_peak = stepnum - warmup_steps
            tail_steps = total_steps - warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592 * prog) + 1.0) * 0.5) * 0.9 + 0.1
        return max(lrmult, 0.1)
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)


def _estimate_chars_per_token(tokenizer_dir, dataset, n_probe=500):
    # Decode a small sample of packed tokens back to text to get chars/token
    # for this tokenizer+corpus pair. Used to label the loss-vs-chars axis.
    tok = HFTokenizer(tokenizer_dir)
    tok.load()
    n_probe = min(n_probe, len(dataset))
    sample = dataset[:n_probe].reshape(-1).tolist()
    text = tok.decode(sample)
    return len(text) / max(len(sample), 1)


def train(args):
    device = pick_device()
    print(f"device: {device}  | seed: {args.seed}")

    _seed_everything(args.seed)

    os.makedirs(args.out, exist_ok=True)

    data = np.load(args.dataset)
    print(f"loaded dataset: {data.shape}  (sequences x (seq_len+1))")

    chars_per_token = _estimate_chars_per_token(args.tokenizer, data)
    print(f"chars/token for this tokenizer+corpus: {chars_per_token:.3f}")

    data_t = torch.tensor(data, dtype=torch.long)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data_t),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_gen,
    )

    tok = HFTokenizer(args.tokenizer)
    tok.load()
    vocab_size = tok.vocab_size

    model = GPTModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        layers=args.layers,
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}  vocab={vocab_size}")

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    max_steps = args.max_steps if args.max_steps > 0 else total_steps

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, args.warmup)
    loss_fn = torch.nn.CrossEntropyLoss()

    checkpoint_steps = set(
        int(s) for s in args.checkpoint_steps.split(",") if s.strip()
    ) if args.checkpoint_steps else set()

    log_rows = []   # (step, tokens_seen, chars_seen, loss)
    tokens_seen = 0
    step = 0

    for epoch in range(args.epochs):
        for (batch,) in loader:
            if step >= max_steps:
                break
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.transpose(1, 2), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            B, S = x.shape
            tokens_seen += B * S

            if step % args.log_interval == 0:
                chars_seen = tokens_seen * chars_per_token
                log_rows.append((step, tokens_seen, chars_seen, loss.item()))
                print(f"step {step:>6} | tokens {tokens_seen:>12,} "
                      f"| chars {int(chars_seen):>14,} | loss {loss.item():.4f}")
                _save_loss_plot(log_rows, os.path.join(args.out, "loss_curve.png"))
                _save_loss_csv(log_rows, os.path.join(args.out, "loss_log.csv"))

            if step in checkpoint_steps:
                ckpt_path = os.path.join(args.out, f"model_weights_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  checkpoint saved at step {step} -> {ckpt_path}")

            step += 1
        if step >= max_steps:
            break

    torch.save(model.state_dict(), os.path.join(args.out, "model_weights.pt"))
    _save_loss_plot(log_rows, os.path.join(args.out, "loss_curve.png"))
    _save_loss_csv(log_rows, os.path.join(args.out, "loss_log.csv"))
    _save_meta(args, vocab_size, n_params, chars_per_token, step,
               os.path.join(args.out, "train_meta.txt"))
    print(f"done. weights + logs in {args.out}")


def _save_loss_plot(rows, path):
    if not rows:
        return
    steps = [r[0] for r in rows]
    tokens = [r[1] for r in rows]
    chars = [r[2] for r in rows]
    losses = [r[3] for r in rows]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(steps, losses)
    ax1.set_xlabel("step"); ax1.set_ylabel("loss"); ax1.set_title("loss vs step")
    ax2.plot(tokens, losses)
    ax2.set_xlabel("tokens"); ax2.set_ylabel("loss"); ax2.set_title("loss vs tokens")
    ax3.plot(chars, losses)
    ax3.set_xlabel("chars"); ax3.set_ylabel("loss"); ax3.set_title("loss vs chars")
    plt.tight_layout()
    plt.savefig(path); plt.close()


def _save_loss_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "tokens_seen", "chars_seen", "loss"])
        w.writerows(rows)


def _save_meta(args, vocab_size, n_params, cpt, final_step, path):
    with open(path, "w") as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"tokenizer: {args.tokenizer}\n")
        f.write(f"vocab_size: {vocab_size}\n")
        f.write(f"params: {n_params}\n")
        f.write(f"d_model: {args.d_model}\n")
        f.write(f"n_heads: {args.n_heads}\n")
        f.write(f"layers: {args.layers}\n")
        f.write(f"seq_len: {args.seq_len}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"warmup: {args.warmup}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"final_step: {final_step}\n")
        f.write(f"chars_per_token: {cpt:.4f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="Cap on total training steps (0 = run --epochs to completion).")
    ap.add_argument("--checkpoint-steps", type=str, default="",
                    help="Comma-separated step numbers at which to save extra weight snapshots.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seeds torch / numpy / DataLoader. Same seed = same run.")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
