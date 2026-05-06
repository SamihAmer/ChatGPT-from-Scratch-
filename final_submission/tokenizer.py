"""
tokenizer.py - thin wrapper around HuggingFace's byte-level BPE.

The same wrapper is used to train every tokenizer in this experiment;
the only thing that varies between tokenizers is the training corpus
(wikitext, PubMed, or MTSamples). Same algorithm, same vocab size,
same merge cap -- the BPE merges that get learned are different
because the input text is different.
"""

import os
from transformers import AutoTokenizer


class HFTokenizer:

    def __init__(self, save_dir):
        # save_dir is where the trained tokenizer files live (or will live).
        self.save_dir = save_dir
        self.tokenizer = None

    def train(self, datafile, vocab_size=10000, limit_alphabet=500):
        # Initialize from GPT-2's tokenizer and retrain it on our corpus.
        base = AutoTokenizer.from_pretrained("gpt2")
        base.eos_token = "<|endoftext|>"

        with open(datafile, "r", encoding="utf-8") as f:
            new = base.train_new_from_iterator(
                f.readlines(),
                vocab_size,
                limit_alphabet=limit_alphabet,
            )

        os.makedirs(self.save_dir, exist_ok=True)
        new.save_pretrained(self.save_dir)
        self.tokenizer = new

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        self.tokenizer.eos_token = "<|endoftext|>"
        # We do our own chunking downstream, so silence the GPT-2 1024-length warning.
        self.tokenizer.model_max_length = int(1e12)

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def encode(self, text):
        return self.tokenizer(text)["input_ids"]

    def decode(self, ids):
        return self.tokenizer.decode(ids)


if __name__ == "__main__":
    # CLI: python tokenizer.py <corpus_txt> <save_dir> [vocab_size]
    import sys
    if len(sys.argv) < 3:
        print("usage: python tokenizer.py <corpus_txt> <save_dir> [vocab_size]")
        sys.exit(1)
    corpus = sys.argv[1]
    save_dir = sys.argv[2]
    vs = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

    tok = HFTokenizer(save_dir)
    tok.train(corpus, vocab_size=vs)
    print(f"Trained tokenizer with vocab={tok.vocab_size} saved to {save_dir}")
