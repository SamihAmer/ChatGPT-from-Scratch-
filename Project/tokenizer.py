'''
HuggingFace byte-level BPE tokenizer wrapper.

Mirrors Module 6's `hftokenizer.py` but:
- Accepts a corpus path and output directory at construction time, so we can
  train and save multiple tokenizers side by side (general vs medical).
- Exposes the HF tokenizer directly for access to things like `eos_token_id`.

Both the baseline and the variation use this same implementation. The only
difference between the two tokenizers is the corpus used to train them.
'''

import os
from transformers import AutoTokenizer


class HFTokenizer:

    def __init__(self, save_dir):
        '''
        save_dir : folder where the trained tokenizer lives (or will be saved to).
        '''
        self.save_dir = save_dir
        self.tokenizer = None  # populated by train() or load()

    def train(self, datafile, vocab_size=10000, limit_alphabet=500):
        '''
        Train a new byte-level BPE tokenizer on the given text file,
        initialized from GPT-2's tokenizer. Saves to self.save_dir.
        '''
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
        # We manage sequence length ourselves via chunking in construct_dataset.py;
        # bump this so HF stops warning about GPT-2's default 1024 limit during encode.
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
