'''
Tokenize a plain-text corpus with a given tokenizer and pack it into
(num_sequences, sequence_length + 1) numpy arrays for training.

Same contract as Module 6's construct_dataset.py, just parameterized:
  - takes a tokenizer path so we can produce one dataset per tokenizer
  - takes an input txt and output npy path

Usage:
  python construct_dataset.py \
      --text artifacts/data/pubmed_train.txt \
      --tokenizer artifacts/tokenizers/medical \
      --out artifacts/datasets/pubmed_train_medical.npy
'''

import argparse
import os
import numpy as np
from tqdm import tqdm

from tokenizer import HFTokenizer


def construct_dataset(text_file, tokenizer_dir, out_path,
                      sequence_length=256, shuffle=True):
    tok = HFTokenizer(tokenizer_dir)
    tok.load()
    eos_id = tok.eos_token_id

    with open(text_file, "r", encoding="utf-8") as f:
        samples = [line.rstrip("\n") for line in f]
    samples = [s for s in samples if s.strip()]

    all_tokens = []
    for s in tqdm(samples, desc=f"tokenize[{os.path.basename(tokenizer_dir)}]"):
        ids = tok.encode(s)
        ids.append(eos_id)
        all_tokens.extend(ids)

    chunk = sequence_length + 1
    num = len(all_tokens) // chunk
    if num == 0:
        raise RuntimeError(f"Too few tokens ({len(all_tokens)}) for seq_len {sequence_length}")
    all_tokens = all_tokens[: num * chunk]

    data = np.array(all_tokens, dtype=np.int32).reshape(num, chunk)
    if shuffle:
        np.random.shuffle(data)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, data)
    print(f"  saved {num:,} sequences of length {chunk} -> {out_path}")
    print(f"  total tokens packed: {num * chunk:,}")
    return num * chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Path to plain-text corpus (one doc/line).")
    ap.add_argument("--tokenizer", required=True, help="Directory of a trained HF tokenizer.")
    ap.add_argument("--out", required=True, help="Output .npy path.")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--no-shuffle", action="store_true")
    args = ap.parse_args()

    construct_dataset(args.text, args.tokenizer, args.out,
                      sequence_length=args.seq_len,
                      shuffle=not args.no_shuffle)


if __name__ == "__main__":
    main()
