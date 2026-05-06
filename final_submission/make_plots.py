"""
make_plots.py - generate the figures referenced in the final report.

Reads loss logs from artifacts/weights/{general_1ep, medical_1ep, general,
medical}/ and writes figures to artifacts/results/figures/. The multi-seed
figures (fig6, fig7) are produced separately by aggregate.py.

Run:  python make_plots.py
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
W = os.path.join(ART, "weights")
FIG = os.path.join(ART, "results", "figures")
os.makedirs(FIG, exist_ok=True)


def read_loss_log(path):
    steps, tokens, chars, losses = [], [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            tokens.append(int(row["tokens_seen"]))
            chars.append(float(row["chars_seen"]))
            losses.append(float(row["loss"]))
    return steps, tokens, chars, losses


# ---------------------------------------------------------------------------
# Figure 1 — token efficiency across corpora
# ---------------------------------------------------------------------------
def fig_token_efficiency():
    # Numbers from token_efficiency.py output.
    corpora = ["pubmed_eval\n(in-domain)", "wikitext_eval\n(out-of-domain)", "mtsamples_eval\n(cross-domain med)"]
    general = [3.82, 4.17, 3.20]
    medical = [4.85, 2.85, 3.21]

    x = np.arange(len(corpora))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width/2, general, width, label="general tokenizer", color="#4c72b0")
    bars2 = ax.bar(x + width/2, medical, width, label="medical tokenizer", color="#c44e52")
    ax.set_ylabel("chars / token  (higher = more compression)")
    ax.set_title("Tokenizer compression on each eval corpus")
    ax.set_xticks(x)
    ax.set_xticklabels(corpora)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(max(general), max(medical)) * 1.15)
    plt.tight_layout()
    out = os.path.join(FIG, "fig1_token_efficiency.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 — 2-epoch training loss across step / tokens / chars
# ---------------------------------------------------------------------------
def fig_training_curves_3axis():
    g_steps, g_tokens, g_chars, g_loss = read_loss_log(os.path.join(W, "general", "loss_log.csv"))
    m_steps, m_tokens, m_chars, m_loss = read_loss_log(os.path.join(W, "medical", "loss_log.csv"))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (gx, mx, xlabel) in zip(
        (ax1, ax2, ax3),
        ((g_steps, m_steps, "step"),
         (g_tokens, m_tokens, "tokens seen"),
         (g_chars, m_chars, "chars seen")),
    ):
        ax.plot(gx, g_loss, label="general", color="#4c72b0", linewidth=1.5)
        ax.plot(mx, m_loss, label="medical", color="#c44e52", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("training loss (nats/token)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax1.set_title("(a) loss vs step\n(same compute)")
    ax2.set_title("(b) loss vs tokens\n(same data volume)")
    ax3.set_title("(c) loss vs chars\n(same text exposure)")
    plt.suptitle("Training loss — 2-epoch runs — across three fairness frames", fontsize=12)
    plt.tight_layout()
    out = os.path.join(FIG, "fig2_training_curves_3axis.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 — 1-epoch vs 2-epoch loss comparison (shows LR-schedule effect)
# ---------------------------------------------------------------------------
def fig_1ep_vs_2ep():
    runs = {
        "general 1-ep": (os.path.join(W, "general_1ep", "loss_log.csv"), "#4c72b0", "--"),
        "general 2-ep": (os.path.join(W, "general",     "loss_log.csv"), "#4c72b0", "-"),
        "medical 1-ep": (os.path.join(W, "medical_1ep", "loss_log.csv"), "#c44e52", "--"),
        "medical 2-ep": (os.path.join(W, "medical",     "loss_log.csv"), "#c44e52", "-"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (path, color, style) in runs.items():
        s, _, _, l = read_loss_log(path)
        ax.plot(s, l, label=label, color=color, linestyle=style, linewidth=1.5, alpha=0.9)
    # mark key boundaries
    ax.axvline(3070, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(3892, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(6141, color="gray", linestyle=":", alpha=0.6)
    ax.text(3070, 9.0, "medical 1-ep end\n/ Opt-1 1-ep", fontsize=8, ha="center", color="gray")
    ax.text(3892, 8.2, "general 1-ep end\n/ Opt-2 1-ep", fontsize=8, ha="center", color="gray")
    ax.text(6141, 9.0, "Opt-1 2-ep\n(medical end)", fontsize=8, ha="center", color="gray")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (nats/token)")
    ax.set_title("1-epoch vs 2-epoch training loss — general's 'plateau' at step ~3000 was LR-schedule-limited, not convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = os.path.join(FIG, "fig3_1ep_vs_2ep.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4 — held-out BPC across checkpoints × corpora × epochs
# ---------------------------------------------------------------------------
def fig_bpc_comparison():
    # all numbers hard-coded from results_{1,2}epoch.md
    corpora = ["pubmed_eval", "wikitext_eval", "mtsamples_eval"]

    # (label, color, values)
    series_1ep = [
        ("general @opt1 (1-ep)", "#8da0cb", [1.2207, 2.7536, 3.0737]),
        ("general @opt2 (1-ep)", "#4c72b0", [1.1926, 2.7664, 3.0756]),
        ("medical       (1-ep)", "#e78ac3", [1.1942, 3.8424, 3.6571]),
    ]
    series_2ep = [
        ("general @opt1 (2-ep)", "#6baed6", [1.0805, 2.7591, 3.0340]),
        ("general @opt2 (2-ep)", "#08519c", [1.0598, 2.7667, 3.0278]),
        ("medical       (2-ep)", "#c44e52", [1.0496, 3.7172, 3.5070]),
    ]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    x = np.arange(len(corpora))
    width = 0.26

    for ax, series, title in [(ax_a, series_1ep, "(a) 1 epoch"),
                               (ax_b, series_2ep, "(b) 2 epochs")]:
        for i, (label, color, vals) in enumerate(series):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=label, color=color)
            for b in bars:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,
                        f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(corpora, fontsize=9)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    ax_a.set_ylabel("BPC (bits/char) — lower is better")
    fig.suptitle("Held-out BPC across corpora, checkpoints, and training horizons", fontsize=12)
    plt.tight_layout()
    out = os.path.join(FIG, "fig4_bpc_comparison.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5 — Δ BPC (1-ep → 2-ep) per model × corpus
# ---------------------------------------------------------------------------
def fig_bpc_delta():
    corpora = ["pubmed_eval", "wikitext_eval", "mtsamples_eval"]
    general_opt2 = [1.1926 - 1.0598, 2.7664 - 2.7667, 3.0756 - 3.0278]
    medical     = [1.1942 - 1.0496, 3.8424 - 3.7172, 3.6571 - 3.5070]

    x = np.arange(len(corpora))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - width/2, general_opt2, width, label="general @opt2", color="#4c72b0")
    b2 = ax.bar(x + width/2, medical,     width, label="medical",       color="#c44e52")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(corpora)
    ax.set_ylabel("BPC improvement (1-ep − 2-ep; higher = bigger gain)")
    ax.set_title("BPC improvement going from 1 epoch to 2 epochs")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + (0.003 if b.get_height() >= 0 else -0.010),
                f"{b.get_height():+.3f}",
                ha="center",
                va="bottom" if b.get_height() >= 0 else "top",
                fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG, "fig5_bpc_delta.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"wrote {out}")


def main():
    fig_token_efficiency()
    fig_training_curves_3axis()
    fig_1ep_vs_2ep()
    fig_bpc_comparison()
    fig_bpc_delta()
    print("\nall figures written to", FIG)


if __name__ == "__main__":
    main()
