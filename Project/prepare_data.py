
'''
Download and prepare the text corpora for training and evaluation.

Outputs (all under artifacts/data/):
  pubmed_train.txt    — PubMed abstracts, training split, one abstract per line
  pubmed_eval.txt     — held-out PubMed abstracts for in-domain eval
  wikitext_train.txt  — wikitext-103 train, used to train the baseline tokenizer
  wikitext_eval.txt   — wikitext-103 validation, general-text eval
  mtsamples_eval.txt  — clinical transcriptions (best-effort). Optional.

All files are plain UTF-8 text with one document per line, matching the
format the course's construct_dataset.py expects.

Usage:
  python prepare_data.py                  # fetch everything with defaults
  python prepare_data.py --pubmed-docs 50000

Tunables at the top of the file control corpus sizes — keep them modest to
avoid spending hours on data prep.
'''

import argparse
import os
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "data")


def _clean(text):
    '''Collapse whitespace and strip, so each doc fits on one line.'''
    if text is None:
        return ""
    return " ".join(text.split())


def _write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            if line:
                f.write(line + "\n")
    print(f"  wrote {len(lines):,} lines -> {path}")


def prepare_pubmed(n_train, n_eval):
    '''
    Pull PubMed abstracts. Tries a short list of Parquet-native HF mirrors
    (modern `datasets` versions no longer support script-based loaders like
    the original `scientific_papers` one).

    Each candidate is (repo, config_or_None, abstract_field, train_split, val_split).
    '''
    candidates = [
        ("ccdv/pubmed-summarization", "document", "abstract", "train", "validation"),
        ("ccdv/pubmed-summarization", "section",  "abstract", "train", "validation"),
        ("armanc/scientific_papers",  "pubmed",   "abstract", "train", "validation"),
    ]
    last_err = None
    for repo, config, field, train_split, val_split in candidates:
        try:
            print(f"[pubmed] trying {repo} (config={config}) ...")
            kwargs = {"split": f"{train_split}[:{n_train}]"}
            if config:
                train_ds = load_dataset(repo, config, **kwargs)
                eval_ds = load_dataset(repo, config,
                                       split=f"{val_split}[:{n_eval}]")
            else:
                train_ds = load_dataset(repo, **kwargs)
                eval_ds = load_dataset(repo, split=f"{val_split}[:{n_eval}]")

            train_lines = [_clean(x[field]) for x in tqdm(train_ds, desc="pubmed-train")]
            eval_lines = [_clean(x[field]) for x in tqdm(eval_ds, desc="pubmed-eval")]

            _write_lines(os.path.join(DATA_DIR, "pubmed_train.txt"), train_lines)
            _write_lines(os.path.join(DATA_DIR, "pubmed_eval.txt"), eval_lines)
            return
        except Exception as e:
            print(f"  failed ({type(e).__name__}: {e})")
            last_err = e

    raise RuntimeError(
        f"All PubMed sources failed. Last error: {last_err}. "
        f"Consider adding another HF Parquet source to the candidates list."
    )


def prepare_wikitext(n_train, n_eval):
    '''
    Pull wikitext-103 train + validation, one paragraph per line.
    Train split is used to train the "general" tokenizer (mirrors Module 6).
    Validation is the general-text eval corpus.
    '''
    print(f"[wikitext] downloading wikitext-103-v1 ...")
    train_ds = load_dataset("wikitext", "wikitext-103-v1",
                            split=f"train[:{n_train}]")
    eval_ds = load_dataset("wikitext", "wikitext-103-v1",
                           split=f"validation[:{n_eval}]")

    train_lines = [_clean(x["text"]) for x in tqdm(train_ds, desc="wiki-train")]
    eval_lines = [_clean(x["text"]) for x in tqdm(eval_ds, desc="wiki-eval")]

    # Filter empties (wikitext has many blank lines)
    train_lines = [l for l in train_lines if l]
    eval_lines = [l for l in eval_lines if l]

    _write_lines(os.path.join(DATA_DIR, "wikitext_train.txt"), train_lines)
    _write_lines(os.path.join(DATA_DIR, "wikitext_eval.txt"), eval_lines)


def prepare_mtsamples():
    '''
    Best-effort MTSamples download. Not guaranteed to succeed — if the HF
    mirror is unavailable the script logs a warning and skips. MTSamples is
    only used as a secondary cross-domain medical eval set, so it's optional.
    '''
    candidates = [
        ("tchebonenko/MedicalTranscriptions", "train", "transcription"),
        ("galactic-sport/mtsamples", "train", "transcription"),
    ]
    for repo, split, field in candidates:
        try:
            print(f"[mtsamples] trying {repo} ...")
            ds = load_dataset(repo, split=split)
            lines = [_clean(x[field]) for x in ds if x.get(field)]
            lines = [l for l in lines if l]
            _write_lines(os.path.join(DATA_DIR, "mtsamples_eval.txt"), lines)
            return
        except Exception as e:
            print(f"  failed ({type(e).__name__}: {e})")
    print("[mtsamples] all sources failed — skipping. You can manually place a "
          "one-doc-per-line file at artifacts/data/mtsamples_eval.txt later.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubmed-train", type=int, default=100000,
                    help="Number of PubMed abstracts for training corpus.")
    ap.add_argument("--pubmed-eval", type=int, default=2000,
                    help="Number of PubMed abstracts for held-out eval.")
    ap.add_argument("--wikitext-train", type=int, default=200000,
                    help="Number of wikitext-103 lines for the general tokenizer's training corpus.")
    ap.add_argument("--wikitext-eval", type=int, default=2000,
                    help="Number of wikitext-103 validation lines for eval.")
    ap.add_argument("--skip-mtsamples", action="store_true")
    args = ap.parse_args()

    prepare_pubmed(args.pubmed_train, args.pubmed_eval)
    prepare_wikitext(args.wikitext_train, args.wikitext_eval)
    if not args.skip_mtsamples:
        prepare_mtsamples()

    print("\nData prep complete. Files in artifacts/data/")


if __name__ == "__main__":
    main()
