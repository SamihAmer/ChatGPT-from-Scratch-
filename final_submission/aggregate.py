"""
aggregate.py - read the multi-seed eval table and produce a summary
report with means/stds and paired t-tests across tokenizers.

Reads:  artifacts/results/eval_table.csv
Writes:
    artifacts/results/aggregate_summary.md   (markdown summary + paired t-tests)
    artifacts/results/aggregate_table.csv    (per-cell mean/std as a tidy CSV)
    artifacts/results/figures/fig6_bpc_with_errorbars.png
    artifacts/results/figures/fig7_bpc_3x3.png

Stats reported:
    - per-(tokenizer, eval_corpus, checkpoint_step) mean/std across seeds
    - paired t-tests (scipy.stats.ttest_rel) pairing seeds between
      tokenizers; 95% CI from the t-interval on paired diffs;
      Cohen's d as mean_diff / std_of_diffs.
"""

import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
R = os.path.join(ART, "results")
FIG = os.path.join(R, "figures")
CSV_IN = os.path.join(R, "eval_table.csv")
SUMMARY_MD = os.path.join(R, "aggregate_summary.md")
SUMMARY_CSV = os.path.join(R, "aggregate_table.csv")
FIG_OUT = os.path.join(FIG, "fig6_bpc_with_errorbars.png")
FIG_3X3 = os.path.join(FIG, "fig7_bpc_3x3.png")

os.makedirs(FIG, exist_ok=True)


def load_rows(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python main.py --phase train_sweep` first."
        )
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["seed"] = int(r["seed"])
            r["checkpoint_step"] = int(r["checkpoint_step"])
            r["bpc"] = float(r["bpc"])
            r["ppl"] = float(r["ppl"])
            r["nll_nats"] = float(r["nll_nats"])
            r["n_tokens"] = int(r["n_tokens"])
            r["n_chars"] = int(r["n_chars"])
            rows.append(r)
    return rows


def per_cell_summary(rows):
    """Group rows by (tokenizer_name, eval_corpus, checkpoint_step) and
    compute mean/std/n of BPC. Returns a list of dicts sorted for stable output."""
    by_cell = defaultdict(list)
    for r in rows:
        key = (r["tokenizer_name"], r["eval_corpus"], r["checkpoint_step"])
        by_cell[key].append(r["bpc"])

    out = []
    for (tok, corp, step), bpcs in sorted(by_cell.items()):
        arr = np.array(bpcs, dtype=float)
        out.append({
            "tokenizer": tok,
            "eval_corpus": corp,
            "checkpoint_step": step,
            "n_seeds": len(arr),
            "bpc_mean": arr.mean(),
            "bpc_std": arr.std(ddof=1) if len(arr) > 1 else 0.0,
            "bpc_min": arr.min(),
            "bpc_max": arr.max(),
        })
    return out


def _paired_diff_stats(a, b, label):
    # a, b are seed-aligned BPC arrays for two tokenizers. Positive mean_diff
    # means tokenizer B wins (lower BPC).
    diff = a - b
    n = len(diff)
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1) if n > 1 else 0.0
    sem_diff = std_diff / np.sqrt(n) if n > 0 else 0.0
    if n > 1 and std_diff > 0:
        ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=sem_diff)
        t_stat, p_value = stats.ttest_rel(a, b)
        cohens_d = mean_diff / std_diff
    else:
        ci_low = ci_high = t_stat = p_value = cohens_d = float("nan")
    return {
        "label": label,
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "tok_a_bpc_mean": a.mean(),
        "tok_b_bpc_mean": b.mean(),
    }


def paired_test_at_step(rows, tok_a, tok_b, eval_corpus, checkpoint_step, label):
    # Match seeds between two tokenizers at a single checkpoint step (used
    # for the matched-compute / Option-1 comparison where both arms are
    # evaluated at the same step number).
    series = {tok_a: {}, tok_b: {}}
    for r in rows:
        if r["eval_corpus"] != eval_corpus:
            continue
        if r["checkpoint_step"] != checkpoint_step:
            continue
        if r["tokenizer_name"] in series:
            series[r["tokenizer_name"]][r["seed"]] = r["bpc"]

    common_seeds = sorted(set(series[tok_a]) & set(series[tok_b]))
    if len(common_seeds) < 2:
        return None

    a = np.array([series[tok_a][s] for s in common_seeds])
    b = np.array([series[tok_b][s] for s in common_seeds])
    out = _paired_diff_stats(a, b, label)
    out["paired_seeds"] = common_seeds
    out["tok_a"] = tok_a
    out["tok_b"] = tok_b
    out["eval_corpus"] = eval_corpus
    out["checkpoint_step_a"] = checkpoint_step
    out["checkpoint_step_b"] = checkpoint_step
    return out


def paired_test_epoch_end(rows, tok_a, tok_b, eval_corpus, label):
    # Match seeds at each tokenizer's own end-of-training checkpoint. The
    # two arms can finish at different step numbers (different dataset
    # sizes), so we can't just match on equal step values.
    best = {}   # (tok, seed) -> (max_step_seen, bpc_at_that_step)
    for r in rows:
        if r["eval_corpus"] != eval_corpus:
            continue
        if r["tokenizer_name"] not in (tok_a, tok_b):
            continue
        key = (r["tokenizer_name"], r["seed"])
        prev = best.get(key)
        if prev is None or r["checkpoint_step"] > prev[0]:
            best[key] = (r["checkpoint_step"], r["bpc"])

    a_seeds = {seed: v for (tok, seed), v in best.items() if tok == tok_a}
    b_seeds = {seed: v for (tok, seed), v in best.items() if tok == tok_b}
    common = sorted(set(a_seeds) & set(b_seeds))
    if len(common) < 2:
        return None

    a = np.array([a_seeds[s][1] for s in common])
    b = np.array([b_seeds[s][1] for s in common])
    out = _paired_diff_stats(a, b, label)
    out["paired_seeds"] = common
    out["tok_a"] = tok_a
    out["tok_b"] = tok_b
    out["eval_corpus"] = eval_corpus
    out["checkpoint_step_a"] = a_seeds[common[0]][0]
    out["checkpoint_step_b"] = b_seeds[common[0]][0]
    return out


def fig_bpc_3x3_matrix(rows, path):
    # 3x3 heatmap: tokenizers down the rows, eval corpora across the columns,
    # using end-of-training checkpoints. Color is per-column normalized so
    # the diagonal-max-by-column pattern reads visually.
    tokenizers = ["general", "medical", "mtsamples"]
    corpora = ["pubmed_eval", "wikitext_eval", "mtsamples_eval"]

    # For each (tokenizer, seed), find the max checkpoint step (i.e. epoch end).
    by_tok_seed = {}
    for r in rows:
        key = (r["tokenizer_name"], r["seed"])
        prev = by_tok_seed.get(key)
        if prev is None or r["checkpoint_step"] > prev:
            by_tok_seed[key] = r["checkpoint_step"]

    cells = {}
    for tok in tokenizers:
        for corp in corpora:
            seed_bpcs = []
            for r in rows:
                if r["tokenizer_name"] != tok or r["eval_corpus"] != corp:
                    continue
                if r["checkpoint_step"] != by_tok_seed[(tok, r["seed"])]:
                    continue
                seed_bpcs.append(r["bpc"])
            arr = np.array(seed_bpcs, dtype=float)
            cells[(tok, corp)] = (arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0, len(arr))

    # Color each cell by per-column rank: 0 = best in column, 1 = worst.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    color_grid = np.zeros((len(tokenizers), len(corpora)))
    for j, corp in enumerate(corpora):
        col_means = np.array([cells[(tok, corp)][0] for tok in tokenizers])
        lo, hi = col_means.min(), col_means.max()
        rng = hi - lo if hi > lo else 1.0
        for i, tok in enumerate(tokenizers):
            color_grid[i, j] = (cells[(tok, corp)][0] - lo) / rng

    im = ax.imshow(color_grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(corpora)))
    ax.set_xticklabels(corpora, fontsize=11)
    ax.set_yticks(range(len(tokenizers)))
    ax.set_yticklabels(tokenizers, fontsize=11)
    ax.set_xlabel("Eval corpus", fontsize=12)
    ax.set_ylabel("Tokenizer (GPT trained with this tokenizer on PubMed)", fontsize=12)
    ax.set_title("Held-out BPC at end-of-training: 3 tokenizers x 3 eval corpora\n"
                 "(mean +/- std across seeds; per-column color = relative rank in that column)",
                 fontsize=11)

    for i, tok in enumerate(tokenizers):
        for j, corp in enumerate(corpora):
            mean, std, n = cells[(tok, corp)]
            txt = f"{mean:.4f}\n+/- {std:.4f}\n(n={n})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("relative rank within column (0=best, 1=worst)", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_summary_csv(per_cell, path):
    fields = ["tokenizer", "eval_corpus", "checkpoint_step", "n_seeds",
              "bpc_mean", "bpc_std", "bpc_min", "bpc_max"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in per_cell:
            w.writerow(row)


def write_summary_md(per_cell, paired_results, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Multi-seed sweep — aggregated results\n\n")

        f.write("## BPC mean ± std across seeds (lower is better)\n\n")
        f.write("| Tokenizer | Eval corpus | Checkpoint | n | Mean BPC | Std | Min | Max |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in per_cell:
            f.write(f"| {row['tokenizer']} | {row['eval_corpus']} | "
                    f"{row['checkpoint_step']} | {row['n_seeds']} | "
                    f"{row['bpc_mean']:.4f} | {row['bpc_std']:.4f} | "
                    f"{row['bpc_min']:.4f} | {row['bpc_max']:.4f} |\n")

        f.write("\n## Paired t-tests across tokenizers (lower BPC = better)\n\n")
        f.write("Pairs are matched by seed (seed 1 of A vs seed 1 of B, etc.). "
                "Positive `mean_diff` means tok_b wins (A's BPC > B's BPC). "
                "Statistical test: `scipy.stats.ttest_rel`; CI: t-interval on paired diffs.\n\n")
        valid = [r for r in paired_results if r is not None]
        if not valid:
            f.write("_No paired comparisons available._\n")
        else:
            f.write("| Comparison | tok_a step | tok_b step | n | Mean (a - b) | 95% CI | t | p | Cohen's d |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in valid:
                f.write(f"| {r['label']} | {r['checkpoint_step_a']} | "
                        f"{r['checkpoint_step_b']} | {r['n']} | "
                        f"{r['mean_diff']:+.4f} BPC | "
                        f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | "
                        f"{r['t_stat']:.3f} | {r['p_value']:.4f} | "
                        f"{r['cohens_d']:+.3f} |\n")

            f.write("\nInterpretation:\n")
            for r in valid:
                winner = r["tok_b"] if r["mean_diff"] > 0 else r["tok_a"]
                sig = "p<0.05 (significant)" if r["p_value"] < 0.05 else "p>=0.05 (not significant)"
                f.write(f"- **{r['label']}**: {winner} wins by "
                        f"{abs(r['mean_diff']):.4f} BPC on average across "
                        f"{r['n']} matched seeds; {sig}; "
                        f"effect size d={r['cohens_d']:+.2f} "
                        f"({r['tok_a']} mean={r['tok_a_bpc_mean']:.4f}, "
                        f"{r['tok_b']} mean={r['tok_b_bpc_mean']:.4f}).\n")


def fig_bpc_errorbars(per_cell, path):
    """Bar chart with error bars: BPC mean across seeds, error = +/- 1 std."""
    corpora = ["pubmed_eval", "wikitext_eval", "mtsamples_eval"]
    tokenizers = ["general", "medical", "mtsamples"]
    colors = {"general": "#4c72b0", "medical": "#c44e52", "mtsamples": "#55a868"}

    # For each (tokenizer, corpus), use the largest checkpoint_step available
    # (= each tokenizer's own end-of-training checkpoint).
    by_tc = defaultdict(list)
    for row in per_cell:
        by_tc[(row["tokenizer"], row["eval_corpus"])].append(row)
    final_per_tc = {
        k: max(rows, key=lambda r: r["checkpoint_step"])
        for k, rows in by_tc.items()
    }

    x = np.arange(len(corpora))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, tok in enumerate(tokenizers):
        means = []
        stds = []
        ns = []
        for corp in corpora:
            row = final_per_tc.get((tok, corp))
            if row is None:
                means.append(np.nan); stds.append(0); ns.append(0)
            else:
                means.append(row["bpc_mean"])
                stds.append(row["bpc_std"])
                ns.append(row["n_seeds"])
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=f"{tok} (n={max(ns) if ns else 0})", color=colors[tok])
        for b, mean, n in zip(bars, means, ns):
            if not np.isnan(mean):
                ax.text(b.get_x() + b.get_width() / 2, mean + 0.05,
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(corpora)
    ax.set_ylabel("BPC (mean ± 1 std across seeds, lower is better)")
    ax.set_title("Held-out BPC at end-of-training, multi-seed sweep")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    rows = load_rows(CSV_IN)
    print(f"loaded {len(rows)} rows from {CSV_IN}")
    if not rows:
        print("nothing to aggregate.")
        return

    per_cell = per_cell_summary(rows)
    write_summary_csv(per_cell, SUMMARY_CSV)
    print(f"wrote {SUMMARY_CSV}")

    # Paired t-tests for the report:
    #   - PubMed (general vs medical) at the matched-compute step (Option 1)
    #     and at each model's epoch end (Option 2)
    #   - wikitext (general vs mtsamples) at epoch end
    #   - MTSamples (general vs mtsamples) and (medical vs mtsamples) at epoch end
    OPT1_STEP = 6141
    paired_results = [
        paired_test_at_step(rows, "general", "medical", "pubmed_eval", OPT1_STEP,
                            label=f"PubMed: general vs medical, Option 1 (matched compute, step {OPT1_STEP})"),
        paired_test_epoch_end(rows, "general", "medical", "pubmed_eval",
                              label="PubMed: general vs medical, Option 2 (epoch end)"),
        paired_test_epoch_end(rows, "general", "mtsamples", "wikitext_eval",
                              label="wikitext: general vs mtsamples, epoch end"),
        paired_test_epoch_end(rows, "medical", "mtsamples", "mtsamples_eval",
                              label="MTSamples: medical vs mtsamples, epoch end"),
        paired_test_epoch_end(rows, "general", "mtsamples", "mtsamples_eval",
                              label="MTSamples: general vs mtsamples, epoch end"),
    ]

    write_summary_md(per_cell, paired_results, SUMMARY_MD)
    print(f"wrote {SUMMARY_MD}")

    fig_bpc_errorbars(per_cell, FIG_OUT)
    print(f"wrote {FIG_OUT}")

    fig_bpc_3x3_matrix(rows, FIG_3X3)
    print(f"wrote {FIG_3X3}")


if __name__ == "__main__":
    main()
