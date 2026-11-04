"""Results figures: drift and harm, main accuracy, cost, held-out clouds (paper Section 5).

Example::  python fig_results.py
"""

import matplotlib.pyplot as plt
import numpy as np
from style import CLOUD, LIGHT, letter, load_stats, save

D = load_stats()


def fig_drift():
    g = D["gate"]["drift"]
    nm = {
        "historical_vs_future": "2010–13 vs 2016–17\n(test period)",
        "historical_vs_adapt": "2010–13 vs 2014–15\n(adaptation period)",
        "historical_split_half_null": "Split-half null\n(no shift)",
    }
    order = [
        "historical_vs_future",
        "historical_vs_adapt",
        "historical_split_half_null",
    ]
    gg = {x["comparison"]: x for x in g}
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.8}
    )
    y = np.arange(3)
    auc = [gg[k]["domain_classifier_roc_auc"] for k in order]
    p = [gg[k]["mmd_permutation_p"] for k in order]
    cols = [CLOUD if pp < 0.05 else LIGHT for pp in p]
    a.hlines(y, 0.5, auc, color=cols, lw=3)
    a.scatter(auc, y, color=cols, s=55, zorder=3, edgecolor="white")
    for k in range(3):
        a.text(
            auc[k] + 0.004,
            k,
            f"{auc[k]:.2f}  (p = {p[k]:.3f})",
            va="center",
            fontsize=8.3,
        )
    a.axvline(0.5, color="#555", ls="--", lw=0.9)
    a.set_yticks(y, [nm[k] for k in order])
    a.set_ylim(2.6, -0.6)
    a.set_xlim(0.49, 0.62)
    a.grid(axis="y", visible=False)
    a.set_xlabel("Domain-classifier ROC AUC (0.5 = chance)")
    a.set_title("Shift between periods")
    letter(a, "a")
    rows = sorted(
        [r for r in D["screens"]],
        key=lambda r: ["dinov2-small", "dinov2-base", "dinov2-large"].index(
            r["backbone"]
        ),
    )
    yy = np.arange(len(rows))
    gap = np.array([100 * float(r["gap"]) for r in rows])
    lo = np.array([100 * float(r["gap_lo"]) for r in rows])
    hi = np.array([100 * float(r["gap_hi"]) for r in rows])
    b.axvspan(0, 5, color=CLOUD, alpha=0.1, lw=0)
    b.errorbar(
        gap,
        yy,
        xerr=[gap - lo, hi - gap],
        fmt="o",
        color="#333",
        ecolor="#888",
        capsize=3,
        ms=5.5,
        lw=1.3,
    )
    b.axvline(0, color="#333", lw=0.9)
    b.text(2.5, 2.4, "harm", color="#A0522D", fontsize=7.5, ha="center", va="center")
    b.set_yticks(yy, ["DINOv2-S", "DINOv2-B", "DINOv2-L"])
    b.set_ylim(2.6, -0.6)
    b.set_xlim(-11, 5)
    b.grid(axis="y", visible=False)
    b.set_xlabel("Old-period val − new-period test accuracy (pp, 95% CI)")
    b.set_title("Harm to the frozen model")
    letter(b, "b")
    save(fig, "fig_drift")


# name, clean, cumulus beside, cumulus over, real clouds, colour, marker (models fine-tuned on clean images;
# heads and LoRA/hybrid are means of seeds 42-44, LoRA r8 is seed 42)
M = [
    ("Frozen baseline", 49.1, 46.8, 42.1, 44.7, "#8C8C8C", "o"),
    ("Learned attention", 60.1, 58.3, 49.4, 55.2, "#8172B3", "o"),
    ("Hard saliency gate", 59.8, 57.4, 48.1, 55.7, "#C44E52", "o"),
    ("Soft saliency gate", 60.3, 58.7, 49.6, 55.6, "#64B5CD", "o"),
    ("LoRA r1, 2 blocks", 58.1, 56.8, 50.6, 55.2, "#937860", "s"),
    ("Hybrid: LoRA r1 + soft gate", 59.8, 58.2, 49.8, 54.9, "#DD8452", "D"),
    ("LoRA r2, 6 blocks", 62.6, 60.5, 53.5, 60.2, "#55A868", "s"),
    ("LoRA r8, 6 blocks", 62.5, 60.7, 52.5, 59.9, "#2E7D4F", "s"),
]


def fig_accuracy():
    tests = [
        "Clean",
        "Cumulus beside object (held out)",
        "Cumulus over object (held out)",
        "Real clouds ≥10% (n = 920)",
    ]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.4, 5.6),
        sharey=True,
        gridspec_kw={"wspace": 0.08, "hspace": 0.42},
    )
    y = np.arange(len(M))
    for t, ax in enumerate(axes.flat):
        values = [m[t + 1] for m in M]
        ax.barh(y, values, height=0.72, color=[m[5] for m in M], zorder=3)
        for k, v in enumerate(values):
            ax.text(v + 0.8, k, f"{v:.1f}", va="center", fontsize=7.2)
        ax.axvline(M[0][t + 1], color="#555", lw=0.9, ls=":", zorder=4)
        ax.set_title(tests[t], fontsize=8.8, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 72)
        ax.set_xlabel("Macro-F1 (%)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_yticks(y, [m[0] for m in M], fontsize=7.8)
    axes[0, 0].set_ylim(len(M) - 0.5, -0.5)
    save(fig, "fig_accuracy")


# name, trainable parameters, backbone weights changed (millions), clean macro-F1, colour, marker
P = [
    ("Attention / saliency heads, width 128", 196930, 0, 60.1, "#8172B3", "o"),
    ("Attention head, width 8", 73807, 0, 60.4, "#B4A7D6", "o"),
    ("LoRA r1, 2 blocks", 36864, 25.2, 58.1, "#937860", "s"),
    ("Hybrid: LoRA r1 + soft gate", 47124, 25.2, 59.8, "#DD8452", "D"),
    ("LoRA r2, 6 blocks", 221184, 75.5, 62.6, "#55A868", "s"),
    ("LoRA r8, 6 blocks", 884736, 75.5, 62.5, "#2E7D4F", "s"),
]


def efficiency_plot(column, xlabel, name, log):
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for label, *values, colour, marker in P:
        ax.scatter(
            values[column],
            values[2],
            color=colour,
            marker=marker,
            s=60,
            zorder=3,
            edgecolor="white",
            lw=0.6,
            label=label,
        )
    if log:
        ax.set_xscale("log")
        ax.set_xlim(2.5e4, 1.5e6)
    else:
        ax.set_xlim(-6, 90)
    ax.set_ylim(56.5, 63.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Clean macro-F1 (%)")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7.8,
        handletextpad=0.4,
        labelspacing=0.9,
    )
    save(fig, name)


def fig_efficiency():
    efficiency_plot(
        0, "Trainable parameters (log scale)", "fig_efficiency_params", True
    )
    efficiency_plot(
        1, "Backbone weights changed (millions)", "fig_efficiency_backbone", False
    )


# macro-F1 fine-tuned with the cloud overlay minus fine-tuned on clean images: beside, over, real clouds (seed 42 for LoRA)
GAIN = [
    ("Learned attention", 58.7 - 58.3, 50.2 - 49.4, 57.1 - 55.2, "#8172B3"),
    ("Hard saliency gate", 57.5 - 57.4, 48.6 - 48.1, 55.8 - 55.7, "#C44E52"),
    ("Soft saliency gate", 58.6 - 58.7, 50.3 - 49.6, 56.0 - 55.6, "#64B5CD"),
    ("LoRA r2, 6 blocks", 58.9 - 60.7, 54.5 - 52.0, 56.6 - 60.1, "#55A868"),
    ("LoRA r8, 6 blocks", 60.6 - 60.7, 56.6 - 52.5, 59.4 - 59.9, "#2E7D4F"),
]


def fig_cloud():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    width, x = 0.16, np.arange(3)
    for k, (label, *gains, colour) in enumerate(GAIN):
        xs = x + (k - 2) * width
        ax.bar(xs, gains, width=width - 0.02, color=colour, label=label, zorder=3)
        for xx, v in zip(xs, gains, strict=True):
            ax.text(
                xx,
                v + (0.12 if v >= 0 else -0.12),
                f"{v:+.1f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=6.6,
            )
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(
        x, ["Cumulus beside object", "Cumulus over object", "Real clouds ≥10%"]
    )
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Change in macro-F1 (points)")
    ax.set_ylim(-4.6, 5.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.8)
    save(fig, "fig_cloud")


def main():
    for make in (fig_drift, fig_accuracy, fig_efficiency, fig_cloud):
        make()


if __name__ == "__main__":
    main()
