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
    # (corpus_file, tokenizer_dir_name)
    # mtsamples tokenizer trains on the same MTSamples corpus that doubles
    # as the cross-domain eval set -- this is a tokenizer-level overlap, not
    # a model-level one (see report Sec 3.1 footnote). The GPT model is
    # never trained on MTSamples text.
    plan = [
        ("wikitext_train.txt",  "general"),
        ("pubmed_train.txt",    "medical"),
        ("mtsamples_eval.txt",  "mtsamples"),
    ]
    for corpus_file, tok_name in plan:
        out_dir = os.path.join(TOK, tok_name)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            print(f"  [skip] tokenizer '{tok_name}' already exists at {out_dir}")
            continue
        run([sys.executable, "tokenizer.py",
             os.path.join(DATA, corpus_file),
             out_dir, "10000"])


def phase_efficiency():
    run([sys.executable, "token_efficiency.py"])


def phase_construct_datasets():
    for name in ("general", "medical", "mtsamples"):
        out_path = os.path.join(DS, f"pubmed_train_{name}.npy")
        if os.path.isfile(out_path):
            print(f"  [skip] dataset {out_path} already exists")
            continue
        run([sys.executable, "construct_dataset.py",
             "--text", os.path.join(DATA, "pubmed_train.txt"),
             "--tokenizer", os.path.join(TOK, name),
             "--out", out_path])


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


def phase_sanity_check():
    """Pre-flight diagnostics on the new mtsamples tokenizer. Run AFTER
    train_tokenizers and construct_datasets, BEFORE train_sweep. Prints:
      - vocab size of all 3 tokenizers
      - chars/token of mtsamples tokenizer on each eval corpus
      - sample tokenizations of medical-academic, clinical-shorthand, and
        general-medical terms

    If anything looks wrong (vocab << 10k, extreme compression in either
    direction, or no merges happening on key terms), STOP and investigate
    before launching the 13-run sweep. Each broken training run wastes
    ~15 minutes."""
    sys.path.insert(0, HERE)
    from tokenizer import HFTokenizer

    print("\n========== sanity check: mtsamples tokenizer ==========\n")

    tok_dirs = {
        "general":   os.path.join(TOK, "general"),
        "medical":   os.path.join(TOK, "medical"),
        "mtsamples": os.path.join(TOK, "mtsamples"),
    }
    toks = {}
    for name, d in tok_dirs.items():
        if not os.path.isdir(d):
            print(f"  [missing] {name} tokenizer at {d} -- run train_tokenizers first")
            return
        t = HFTokenizer(d)
        t.load()
        toks[name] = t
        print(f"  {name:9s}  vocab_size = {t.vocab_size}")

    print("\n--- chars/token on each eval corpus ---")
    print(f"  {'corpus':<20s} {'general':>10s} {'medical':>10s} {'mtsamples':>10s}")
    for corpus in ("pubmed_eval", "wikitext_eval", "mtsamples_eval"):
        path = os.path.join(DATA, f"{corpus}.txt")
        if not os.path.isfile(path):
            print(f"  {corpus:<20s} (file missing, skipping)")
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        n_chars = len(text)
        ratios = []
        for name in ("general", "medical", "mtsamples"):
            n_tok = len(toks[name].encode(text))
            ratios.append(n_chars / max(n_tok, 1))
        print(f"  {corpus:<20s} {ratios[0]:>10.3f} {ratios[1]:>10.3f} {ratios[2]:>10.3f}")

    print("\n--- sample tokenizations ---")
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

    print("\n========== sanity check complete ==========")
    print("If mtsamples chars/token looks plausible (similar magnitude to medical")
    print("on PubMed, comparable to general on wikitext) and the sample splits")
    print("show meaningful merges (not pure char-by-char), proceed with train_sweep.")
    print("If anything looks off, investigate BEFORE burning 13 training runs.\n")


# Multi-seed sweep: which seeds to run for each tokenizer.
SWEEP_PLAN = [
    ("general",   [1, 2, 3, 4, 5]),
    ("medical",   [1, 2, 3, 4, 5]),
    ("mtsamples", [1, 2, 3]),
]
# Hard-coded matched-compute boundary (medical's 2-epoch endpoint). Do NOT
# recompute as min across tokenizers -- that would redefine the comparison
# against the existing report.
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
    """Multi-seed sweep over the 3 tokenizers. For each (tokenizer, seed):
      1. Train 2 epochs with the given seed and a step-OPT1_STEP checkpoint.
      2. Immediately evaluate the final and (if reached) opt1 checkpoints
         on all 3 eval corpora, appending rows to artifacts/results/eval_table.csv.

    Inline eval is intentional: if the sweep crashes at run K, runs 1..K-1
    are already in the CSV. Skipped checkpoints (when opt1_step exceeds a
    short run's total) are silently omitted from the CSV with a console note."""
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
            print(f"\n========== sweep run {run_idx}/{total}: "
                  f"{tokenizer_name} seed={seed} ==========")

            # --- 1. Train ---
            run([sys.executable, "train.py",
                 "--dataset", dataset_npy,
                 "--tokenizer", tok_dir,
                 "--out", out_dir,
                 "--epochs", "2",
                 "--seed", str(seed),
                 "--checkpoint-steps", str(OPT1_STEP)])

            # --- 2. Inline eval on every (checkpoint, corpus) combo ---
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
                    print(f"  [skip-eval] no checkpoint at {ckpt_path} "
                          f"(this run did not reach step {ckpt_step})")
                    continue
                for eval_corpus in eval_corpora:
                    text_path = os.path.join(DATA, f"{eval_corpus}.txt")
                    if not os.path.isfile(text_path):
                        continue
                    print(f"  [eval] {tokenizer_name} seed={seed} "
                          f"step={ckpt_step} on {eval_corpus}")
                    run([sys.executable, "evaluate.py",
                         "--weights", ckpt_path,
                         "--tokenizer", tok_dir,
                         "--text", text_path,
                         "--csv-append", csv_path,
                         "--seed", str(seed),
                         "--tokenizer-name", tokenizer_name,
                         "--checkpoint-step", str(ckpt_step),
                         "--eval-corpus", eval_corpus])

    print(f"\n========== sweep complete ({total} runs) ==========")
    print(f"results -> {csv_path}")
    print(f"next:  python aggregate.py")


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
    "sanity_check": phase_sanity_check,
    "train_all": phase_train_all,
    "train_sweep": phase_train_sweep,
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
