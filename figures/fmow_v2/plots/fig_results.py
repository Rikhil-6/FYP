"""Results figures: drift and harm, main accuracy, cost, held-out clouds (paper Section 5).

Example::  python fig_results.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from style import CLOUD, LIGHT, letter, load_stats, save

D = load_stats()


def fig_drift():
    g = D["gate"]["drift"]
    nm = {
        "historical_vs_future": "2010–13 vs 2016–17\n(test period)",
        "historical_vs_adapt": "2010–13 vs 2014–15\n(adaptation period)",
        "historical_split_half_null": "Split-half null\n(no shift)",
    }
    order = ["historical_vs_future", "historical_vs_adapt", "historical_split_half_null"]
    gg = {x["comparison"]: x for x in g}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.8})
    y = np.arange(3)
    auc = [gg[k]["domain_classifier_roc_auc"] for k in order]
    p = [gg[k]["mmd_permutation_p"] for k in order]
    cols = [CLOUD if pp < 0.05 else LIGHT for pp in p]
    a.hlines(y, 0.5, auc, color=cols, lw=3)
    a.scatter(auc, y, color=cols, s=55, zorder=3, edgecolor="white")
    for k in range(3):
        a.text(auc[k] + 0.004, k, f"{auc[k]:.2f}  (p = {p[k]:.3f})", va="center", fontsize=8.3)
    a.axvline(0.5, color="#555", ls="--", lw=0.9)
    a.set_yticks(y, [nm[k] for k in order])
    a.set_ylim(2.6, -0.6)
    a.set_xlim(0.49, 0.62)
    a.grid(axis="y", visible=False)
    a.set_xlabel("Domain-classifier ROC AUC (0.5 = chance)")
    a.set_title("Shift between periods")
    letter(a, "a")
    rows = sorted(
        [r for r in D["screens"]], key=lambda r: ["dinov2-small", "dinov2-base", "dinov2-large"].index(r["backbone"])
    )
    yy = np.arange(len(rows))
    gap = np.array([100 * float(r["gap"]) for r in rows])
    lo = np.array([100 * float(r["gap_lo"]) for r in rows])
    hi = np.array([100 * float(r["gap_hi"]) for r in rows])
    b.axvspan(0, 5, color=CLOUD, alpha=0.1, lw=0)
    b.errorbar(gap, yy, xerr=[gap - lo, hi - gap], fmt="o", color="#333", ecolor="#888", capsize=3, ms=5.5, lw=1.3)
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


M = [
    ("Frozen baseline", 49.1, 41.9, 46.8, 42.1, 44.7, "#8C8C8C", "o"),
    ("Learned attention", 60.1, None, 58.3, 49.4, 55.2, "#8172B3", "o"),
    ("Hard saliency gate", 59.8, None, 57.4, 48.1, 55.7, "#C44E52", "o"),
    ("Soft saliency gate", 60.3, None, 58.7, 49.6, 55.6, "#64B5CD", "o"),
    ("LoRA r1, 2 blocks", 57.7, 50.1, 56.2, 50.9, 56.4, "#937860", "s"),
    ("Hybrid: LoRA r1, 2 blocks\n+ soft gate", 60.0, 50.2, 58.7, 49.9, 54.8, "#DD8452", "D"),
    ("LoRA r2, 6 blocks", 62.7, 54.1, 60.7, 52.0, 60.1, "#55A868", "s"),
    ("LoRA r8, 6 blocks", 62.5, 53.9, 60.7, 52.5, 59.9, "#2E7D4F", "s"),
]


def fig_accuracy():
    tests = [
        (1, "Clean"),
        (3, "Cumulus beside object\n(held out)"),
        (4, "Cumulus over object\n(held out)"),
        (5, "Real clouds ≥10%\n(n = 920)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.3), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(M))
    for a, (ti, tl) in zip(axes, tests, strict=True):
        for k, m in enumerate(M):
            a.scatter(m[ti], k, color=m[6], marker=m[7], s=42, zorder=3, edgecolor="white", lw=0.6)
            a.text(m[ti] + 0.5, k, f"{m[ti]:.1f}", va="center", fontsize=6.8)
        a.set_title(tl, fontsize=8.2, loc="center")
        a.grid(axis="y", visible=False)
        a.set_xlim(40, 66)
        a.axvline(M[0][ti], color="#bbb", lw=0.8, ls=":")
    axes[0].set_yticks(y, [m[0] for m in M], fontsize=7.6)
    axes[0].set_ylim(len(M) - 0.5, -0.5)
    fig.supxlabel("Macro-F1 (%) on 2016–17 test images, models trained on clean images", fontsize=8, y=-0.02)
    save(fig, "fig_accuracy")


P = [
    ("Learned attention", 196929, 0, 60.1, 49.4, "#8172B3", "o"),
    ("Hard gate", 196931, 0, 59.8, 48.1, "#C44E52", "o"),
    ("Soft gate", 196932, 0, 60.3, 49.6, "#64B5CD", "o"),
    ("Attention, width 8", 73807, 0, 60.4, None, "#8172B3", "o"),
    ("LoRA r1, 2 blocks", 36864, 25.2, 57.7, 50.9, "#937860", "s"),
    ("Hybrid", 47124, 25.2, 60.0, 49.9, "#DD8452", "D"),
    ("LoRA r2, 6 blocks", 221184, 75.5, 62.7, 52.0, "#55A868", "s"),
    ("LoRA r8, 6 blocks", 884736, 75.5, 62.5, 52.5, "#2E7D4F", "s"),
]


def fig_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"wspace": 0.32})
    for a, xi, xl, log in (
        (axes[0], 1, "Trainable parameters (log scale)", True),
        (axes[1], 2, "Backbone weights changed (millions)", False),
    ):
        for n, *v, c, mk in P:
            x = v[xi - 1]
            a.scatter(x, v[2], color=c, marker=mk, s=48, zorder=3, edgecolor="white", lw=0.6)
            off = {
                "LoRA r8, 6 blocks": (6, -8),
                "LoRA r2, 6 blocks": (6, 6),
                "Attention, width 8": (0, 10),
                "Learned attention": (8, -14),
            }.get(n, (7, 0))
            ha = "center" if n == "Attention, width 8" else "left"
            if n in ("Hard gate", "Soft gate") or (xi == 2 and n == "Attention, width 8"):
                continue
            lab = (
                ("Heads (width 128): attention,\nsoft gate, hard gate" if xi == 1 else "Heads, all widths")
                if n == "Learned attention"
                else n
            )
            if xi == 2 and n.startswith("LoRA r"):
                off = (-6, 8 if "r2" in n else -9)
                ha = "right"
            a.annotate(lab, (x, v[2]), xytext=off, textcoords="offset points", fontsize=6.8, ha=ha, va="center")
        a.set_xlabel(xl, fontsize=8)
        a.set_ylim(56.5, 63.5)
        if log:
            a.set_xscale("log")
            a.set_xlim(25000.0, 1500000.0)
        else:
            a.set_xlim(-6, 90)
    axes[0].set_ylabel("Clean macro-F1 (%)")
    axes[0].set_title("Accuracy vs trainable parameters")
    axes[1].set_title("Accuracy vs backbone change")
    letter(axes[0], "a")
    letter(axes[1], "b")
    save(fig, "fig_efficiency")


def fig_cloud():
    rows = [
        ("Learned attention", 58.7, 50.2, 57.1, 55.8, "#8172B3", "o"),
        ("Hard saliency gate", 57.5, 48.6, 55.8, 55.5, "#C44E52", "o"),
        ("Soft saliency gate", 58.6, 50.3, 56.0, 56.3, "#64B5CD", "o"),
        ("LoRA r2, 6 blocks", 58.9, 54.5, 56.6, 57.8, "#55A868", "s"),
        ("LoRA r8, 6 blocks", 60.6, 56.6, 59.4, 59.2, "#2E7D4F", "s"),
    ]
    clean_tr = {
        "Learned attention": (58.3, 49.4, 55.2),
        "Hard saliency gate": (57.4, 48.1, 55.7),
        "Soft saliency gate": (58.7, 49.6, 55.6),
        "LoRA r2, 6 blocks": (60.7, 52.0, 60.1),
        "LoRA r8, 6 blocks": (60.7, 52.5, 59.9),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True, gridspec_kw={"wspace": 0.1})
    y = np.arange(len(rows))
    for a, (ci, ttl) in zip(
        axes, ((1, "Cumulus beside object"), (2, "Cumulus over object"), (3, "Real clouds ≥10%")), strict=True
    ):
        for k, r in enumerate(rows):
            c0 = clean_tr[r[0]][ci - 1]
            c1 = r[ci]
            a.plot([c0, c1], [k, k], color="#ccc", lw=2, zorder=1)
            a.scatter(c0, k, facecolor="white", edgecolor=r[5], marker=r[6], s=40, zorder=3, lw=1.3)
            a.scatter(c1, k, color=r[5], marker=r[6], s=40, zorder=3, edgecolor="white", lw=0.6)
        a.set_title(ttl, fontsize=8.5, loc="center")
        a.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [r[0] for r in rows], fontsize=7.6)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    axes[1].legend(
        [Line2D([], [], marker="o", ls="", mfc="white", mec="#555"), Line2D([], [], marker="o", ls="", color="#555")],
        ["Trained on clean images", "Trained with cloud overlay"],
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
    )
    fig.supxlabel("Macro-F1 (%)", fontsize=8, y=-0.03)
    save(fig, "fig_cloud")


def main():
    for make in (fig_drift, fig_accuracy, fig_efficiency, fig_cloud):
        make()


if __name__ == "__main__":
    main()
