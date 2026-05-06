"""
generate.py - sample text continuations from a trained GPT, using the
sampler from sampler.py. Outputs a single text file with one section per
prompt so we can compare two models side by side in the report.

Example:
    python generate.py --weights artifacts/weights/medical/model_weights.pt \\
                       --tokenizer artifacts/tokenizers/medical \\
                       --label medical \\
                       --out artifacts/results/samples_medical.txt
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from gpt import GPTModel
from tokenizer import HFTokenizer
from sampler import Sampler
from train import pick_device


PROMPTS = [
    "Patient presents with",
    "CT of the abdomen shows",
    "The patient was administered",
    "Postoperative diagnosis:",
    "Laparoscopic cholecystectomy was performed",
    "The quick brown fox",   # non-medical sanity check
]


def generate(weights_path, tokenizer_dir, out_path, label,
             n_tokens=120, d_model=512, n_heads=8, layers=6, seq_len=256,
             top_p=0.8, freq_penalty=1.1, presence_penalty=1.1):
    device = pick_device()

    tok = HFTokenizer(tokenizer_dir)
    tok.load()
    vocab_size = tok.vocab_size

    model = GPTModel(d_model=d_model, n_heads=n_heads, layers=layers,
                     vocab_size=vocab_size, max_seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    samp = Sampler(top_p=top_p, frequency_penalty=freq_penalty,
                   presence_penalty=presence_penalty)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [f"# Generations from [{label}]  weights={weights_path}  tokenizer={tokenizer_dir}\n"]

    for prompt in PROMPTS:
        ids = tok.encode(prompt)
        ids = torch.tensor([ids], device=device)

        for _ in tqdm(range(n_tokens), desc=f"gen[{label}] {prompt[:30]!r}"):
            with torch.no_grad():
                logits = model(ids)
            logits = logits[0, -1, :].detach().cpu().numpy()
            prev = ids[0].detach().cpu().numpy()
            next_id = int(samp(logits, prev))
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == tok.eos_token_id:
                break
            if ids.shape[1] >= seq_len:
                break

        text = tok.decode(ids[0].detach().cpu().numpy())
        lines.append(f"\n--- prompt: {prompt!r} ---\n{text}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--label", required=True, help="Short name printed in the output header.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tokens", type=int, default=120)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()

    generate(args.weights, args.tokenizer, args.out, args.label,
             n_tokens=args.n_tokens, d_model=args.d_model,
             n_heads=args.n_heads, layers=args.layers, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
