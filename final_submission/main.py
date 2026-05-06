"""
main.py - top-level driver for the medical BPE tokenizer experiment.

The full pipeline is split into phases. Run them one at a time:

    python main.py --phase prepare_data
    python main.py --phase train_tokenizers
    python main.py --phase efficiency
    python main.py --phase construct_datasets
    python main.py --phase sanity_check     # quick check of the mtsamples tokenizer
    python main.py --phase train_sweep      # multi-seed sweep, slow
    python aggregate.py                     # multi-seed analysis (separate script)

Or chain them all:  python main.py --phase all

train_sweep is the only slow phase (~3.7 hr on a 4070 Ti for 13 runs).
The other phases finish in seconds-to-minutes.
"""

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
    # Three BPE tokenizers, each trained on a different corpus. Same algorithm
    # and vocab size (10k) for all three; only the training data differs.
    plan = [
        ("wikitext_train.txt",  "general"),
        ("pubmed_train.txt",    "medical"),
        ("mtsamples_eval.txt",  "mtsamples"),
    ]
    for corpus_file, tok_name in plan:
        out_dir = os.path.join(TOK, tok_name)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            print(f"  [skip] tokenizer '{tok_name}' already exists")
            continue
        run([sys.executable, "tokenizer.py",
             os.path.join(DATA, corpus_file),
             out_dir, "10000"])


def phase_efficiency():
    run([sys.executable, "token_efficiency.py"])


def phase_construct_datasets():
    # Pack PubMed under each of the 3 tokenizers (the GPT trains on PubMed
    # regardless of which tokenizer is used).
    for name in ("general", "medical", "mtsamples"):
        out_path = os.path.join(DS, f"pubmed_train_{name}.npy")
        if os.path.isfile(out_path):
            print(f"  [skip] {out_path} already exists")
            continue
        run([sys.executable, "construct_dataset.py",
             "--text", os.path.join(DATA, "pubmed_train.txt"),
             "--tokenizer", os.path.join(TOK, name),
             "--out", out_path])


def phase_smoke():
    # 100-step training run to verify the loop works end to end before
    # committing to a long sweep.
    run([sys.executable, "train.py",
         "--dataset", os.path.join(DS, "pubmed_train_medical.npy"),
         "--tokenizer", os.path.join(TOK, "medical"),
         "--out", os.path.join(W, "smoke"),
         "--max-steps", "100",
         "--log-interval", "10"])


def phase_train_all():
    # Single-run training of general + medical (no seed sweep). Saves a
    # checkpoint at step 6141 for the matched-compute (Option-1) comparison.
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


def phase_sanity_check():
    # Quick diagnostics on the mtsamples tokenizer before kicking off the
    # full sweep. Prints chars/token on each eval corpus and shows how the
    # three tokenizers split a few representative medical/clinical phrases.
    sys.path.insert(0, HERE)
    from tokenizer import HFTokenizer

    print("\n--- mtsamples tokenizer sanity check ---\n")

    tok_dirs = {
        "general":   os.path.join(TOK, "general"),
        "medical":   os.path.join(TOK, "medical"),
        "mtsamples": os.path.join(TOK, "mtsamples"),
    }
    toks = {}
    for name, d in tok_dirs.items():
        if not os.path.isdir(d):
            print(f"  missing: {name} tokenizer at {d}")
            return
        t = HFTokenizer(d)
        t.load()
        toks[name] = t
        print(f"  {name:9s}  vocab_size = {t.vocab_size}")

    print("\nchars/token on each eval corpus:")
    print(f"  {'corpus':<20s} {'general':>10s} {'medical':>10s} {'mtsamples':>10s}")
    for corpus in ("pubmed_eval", "wikitext_eval", "mtsamples_eval"):
        path = os.path.join(DATA, f"{corpus}.txt")
        if not os.path.isfile(path):
            print(f"  {corpus:<20s} (missing)")
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        n_chars = len(text)
        ratios = []
        for name in ("general", "medical", "mtsamples"):
            n_tok = len(toks[name].encode(text))
            ratios.append(n_chars / max(n_tok, 1))
        print(f"  {corpus:<20s} {ratios[0]:>10.3f} {ratios[1]:>10.3f} {ratios[2]:>10.3f}")

    print("\nsample tokenizations:")
    samples = [
        "cholecystectomy",
        "myocardial infarction",
        "pt y/o w/",
        "laparoscopic",
        "Patient was admitted",
    ]
    for s in samples:
        print(f"\n  {s!r}")
        for name in ("general", "medical", "mtsamples"):
            ids = toks[name].encode(s)
            pieces = [toks[name].tokenizer.decode([tid]) for tid in ids]
            print(f"    {name:9s} ({len(ids):>2d}): {' | '.join(repr(p) for p in pieces)}")


# Sweep configuration. Seeds 1-5 for general/medical; mtsamples is a
# secondary arm so 3 seeds is enough for a directional claim.
SWEEP_PLAN = [
    ("general",   [1, 2, 3, 4, 5]),
    ("medical",   [1, 2, 3, 4, 5]),
    ("mtsamples", [1, 2, 3]),
]
# Matched-compute boundary = medical's 2-epoch endpoint (step 6141).
# This is the step number used for the Option-1 fairness frame.
OPT1_STEP = 6141


def _read_meta(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def phase_train_sweep():
    # 13 training runs (5 + 5 + 3) with inline eval after each one.
    # Inline eval is intentional: if the sweep crashes at run K, runs 1..K-1
    # already have their numbers in the CSV.
    csv_path = os.path.join(R, "eval_table.csv")
    os.makedirs(R, exist_ok=True)

    eval_corpora = ("pubmed_eval", "wikitext_eval", "mtsamples_eval")

    total = sum(len(seeds) for _, seeds in SWEEP_PLAN)
    run_idx = 0
    for tokenizer_name, seeds in SWEEP_PLAN:
        dataset_npy = os.path.join(DS, f"pubmed_train_{tokenizer_name}.npy")
        tok_dir = os.path.join(TOK, tokenizer_name)
        for seed in seeds:
            run_idx += 1
            out_dir = os.path.join(W, f"{tokenizer_name}_seed{seed}")
            print(f"\n--- sweep run {run_idx}/{total}: {tokenizer_name} seed={seed} ---")

            # train
            run([sys.executable, "train.py",
                 "--dataset", dataset_npy,
                 "--tokenizer", tok_dir,
                 "--out", out_dir,
                 "--epochs", "2",
                 "--seed", str(seed),
                 "--checkpoint-steps", str(OPT1_STEP)])

            # eval at end-of-training and at the matched-compute boundary
            meta = _read_meta(os.path.join(out_dir, "train_meta.txt"))
            try:
                final_step = int(meta.get("final_step", -1))
            except ValueError:
                final_step = -1

            ckpts = [
                ("model_weights.pt",                 final_step),
                (f"model_weights_step{OPT1_STEP}.pt", OPT1_STEP),
            ]
            for ckpt_file, ckpt_step in ckpts:
                ckpt_path = os.path.join(out_dir, ckpt_file)
                if not os.path.isfile(ckpt_path):
                    print(f"  [skip-eval] no checkpoint at {ckpt_path}")
                    continue
                for eval_corpus in eval_corpora:
                    text_path = os.path.join(DATA, f"{eval_corpus}.txt")
                    if not os.path.isfile(text_path):
                        continue
                    print(f"  [eval] {tokenizer_name} seed={seed} step={ckpt_step} on {eval_corpus}")
                    run([sys.executable, "evaluate.py",
                         "--weights", ckpt_path,
                         "--tokenizer", tok_dir,
                         "--text", text_path,
                         "--csv-append", csv_path,
                         "--seed", str(seed),
                         "--tokenizer-name", tokenizer_name,
                         "--checkpoint-step", str(ckpt_step),
                         "--eval-corpus", eval_corpus])

    print(f"\nsweep done ({total} runs). results -> {csv_path}")
    print("next: python aggregate.py")


def phase_evaluate():
    # Single-run eval (used after train_all). For multi-seed eval,
    # train_sweep handles it inline.
    runs = [
        ("general", "model_weights.pt",            "general_2ep"),
        ("general", "model_weights_step6141.pt",   "general_step6141"),
        ("medical", "model_weights.pt",            "medical_2ep"),
    ]
    for name, weights_file, label in runs:
        weights = os.path.join(W, name, weights_file)
        if not os.path.isfile(weights):
            print(f"skip evaluate[{label}]: no weights at {weights}")
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
            print(f"skip generate[{name}]: no weights at {weights}")
            continue
        run([sys.executable, "generate.py",
             "--weights", weights,
             "--tokenizer", os.path.join(TOK, name),
             "--label", name,
             "--out", os.path.join(R, f"samples_{name}.txt")])


PHASES = {
    "prepare_data":       phase_prepare_data,
    "train_tokenizers":   phase_train_tokenizers,
    "efficiency":         phase_efficiency,
    "construct_datasets": phase_construct_datasets,
    "smoke":              phase_smoke,
    "sanity_check":       phase_sanity_check,
    "train_all":          phase_train_all,
    "train_sweep":        phase_train_sweep,
    "evaluate":           phase_evaluate,
    "generate":           phase_generate,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=list(PHASES.keys()) + ["all"])
    args = ap.parse_args()

    if args.phase == "all":
        for name, fn in PHASES.items():
            print(f"\n========== phase: {name} ==========")
            fn()
    else:
        PHASES[args.phase]()


if __name__ == "__main__":
    main()
