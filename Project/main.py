'''
End-to-end orchestrator for the medical tokenizer project.

Runs the pipeline in phases so you can pause at any step:
  1. prepare_data       — download PubMed / wikitext / (try) MTSamples
  2. train_tokenizers   — train the general and medical HF BPE tokenizers
  3. efficiency         — token efficiency analysis (no GPT needed)
  4. construct_datasets — pack the PubMed training corpus under each tokenizer
  5. smoke              — 100-step training smoke test on the medical dataset
  6. train_all          — full training of both GPT models (slow; CUDA recommended)
  7. evaluate           — perplexity + BPC on held-out corpora
  8. generate           — qualitative side-by-side generation samples

Usage:
  python main.py --phase prepare_data
  python main.py --phase efficiency
  python main.py --phase smoke
  python main.py --phase all            # runs every phase end to end

Training (phases 5, 6) is the only slow part. The rest finishes in minutes.
'''

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DATA = os.path.join(ART, "data")
TOK = os.path.join(ART, "tokenizers")
DS = os.path.join(ART, "datasets")
W = os.path.join(ART, "weights")
R = os.path.join(ART, "results")


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=HERE)
    if res.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}")


def phase_prepare_data():
    run([sys.executable, "prepare_data.py"])


def phase_train_tokenizers():
    run([sys.executable, "tokenizer.py",
         os.path.join(DATA, "wikitext_train.txt"),
         os.path.join(TOK, "general"), "10000"])
    run([sys.executable, "tokenizer.py",
         os.path.join(DATA, "pubmed_train.txt"),
         os.path.join(TOK, "medical"), "10000"])


def phase_efficiency():
    run([sys.executable, "token_efficiency.py"])


def phase_construct_datasets():
    for name in ("general", "medical"):
        run([sys.executable, "construct_dataset.py",
             "--text", os.path.join(DATA, "pubmed_train.txt"),
             "--tokenizer", os.path.join(TOK, name),
             "--out", os.path.join(DS, f"pubmed_train_{name}.npy")])


def phase_smoke():
    run([sys.executable, "train.py",
         "--dataset", os.path.join(DS, "pubmed_train_medical.npy"),
         "--tokenizer", os.path.join(TOK, "medical"),
         "--out", os.path.join(W, "smoke"),
         "--max-steps", "100",
         "--log-interval", "10"])


def phase_train_all():
    # We train both models for 2 epochs with a cosine LR schedule spanning the
    # full 2-epoch horizon. Medical has 3,071 batches/epoch, so its 2-epoch
    # endpoint is step 6141 (indices 0..6141 = 6142 optimizer updates, the last
    # batch of each epoch is partial). General has 3,892 batches/epoch, so its
    # 2-epoch endpoint is step 7783. We save a matched-compute checkpoint for
    # general at step 6141 to enable the same-step (Option 1) BPC comparison.
    EPOCHS = "2"
    SAME_STEP_BOUNDARY = "6141"
    for name in ("general", "medical"):
        cmd = [sys.executable, "train.py",
               "--dataset", os.path.join(DS, f"pubmed_train_{name}.npy"),
               "--tokenizer", os.path.join(TOK, name),
               "--out", os.path.join(W, name),
               "--epochs", EPOCHS]
        if name == "general":
            cmd += ["--checkpoint-steps", SAME_STEP_BOUNDARY]
        run(cmd)


def phase_evaluate():
    # (tokenizer_name, weights_filename, label)
    # The step6141 checkpoint is general's compute-matched snapshot for the
    # 2-epoch run so we can compare general@6141 against medical@6141
    # (medical's 2-epoch end-of-training).
    runs = [
        ("general", "model_weights.pt",            "general_2ep"),
        ("general", "model_weights_step6141.pt",   "general_step6141"),
        ("medical", "model_weights.pt",            "medical_2ep"),
    ]
    for name, weights_file, label in runs:
        weights = os.path.join(W, name, weights_file)
        if not os.path.isfile(weights):
            print(f"skipping evaluate[{label}]: no weights at {weights}")
            continue
        for eval_corpus in ("pubmed_eval", "wikitext_eval", "mtsamples_eval"):
            path = os.path.join(DATA, f"{eval_corpus}.txt")
            if not os.path.isfile(path):
                continue
            print(f"\n--- {label} on {eval_corpus} ---")
            run([sys.executable, "evaluate.py",
                 "--weights", weights,
                 "--tokenizer", os.path.join(TOK, name),
                 "--text", path])


def phase_generate():
    for name in ("general", "medical"):
        weights = os.path.join(W, name, "model_weights.pt")
        if not os.path.isfile(weights):
            print(f"skipping generate[{name}]: no weights at {weights}")
            continue
        run([sys.executable, "generate.py",
             "--weights", weights,
             "--tokenizer", os.path.join(TOK, name),
             "--label", name,
             "--out", os.path.join(R, f"samples_{name}.txt")])


PHASES = {
    "prepare_data": phase_prepare_data,
    "train_tokenizers": phase_train_tokenizers,
    "efficiency": phase_efficiency,
    "construct_datasets": phase_construct_datasets,
    "smoke": phase_smoke,
    "train_all": phase_train_all,
    "evaluate": phase_evaluate,
    "generate": phase_generate,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=list(PHASES.keys()) + ["all"])
    args = ap.parse_args()

    if args.phase == "all":
        for name, fn in PHASES.items():
            print(f"\n========== phase: {name} ==========")
            fn()
    else:
        PHASES[args.phase]()


if __name__ == "__main__":
    main()
