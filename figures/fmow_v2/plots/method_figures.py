"""Learned gate response curves (paper Figure 3); the schematics are TikZ sources in ../tikz.

Example::  python method_figures.py
"""

import matplotlib.pyplot as plt
import numpy as np

from style import load_stats, save

TRAIN_E, SAL_E = "#C46A2F", "#4C72B0"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gate_curves(hard, soft):
    """Share of learned weight kept vs saliency, for the learned hard gate (gamma, delta) and soft gate (gamma, delta, rho)."""
    z = np.linspace(-3, 3, 300)
    g_hard = sigmoid(hard[0] * z + hard[1])
    g_soft = soft[2] + (1 - soft[2]) * sigmoid(soft[0] * z + soft[1])
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.axvspan(-3, -1, color="#F3F3F3", lw=0)
    ax.plot(z, np.ones_like(z), color="#8C8C8C", lw=1.4, ls=":", label="Learned attention (no gate)")
    ax.plot(z, g_soft, color=SAL_E, lw=1.8, ls="--", label=f"Soft gate (floor ρ = {soft[2]:.2f})")
    ax.plot(z, g_hard, color=TRAIN_E, lw=1.8, label=f"Hard gate (γ = {hard[0]:.2f}, δ = {hard[1]:.2f})")
    ax.text(-2.0, 0.08, "low-evidence\npatches", ha="center", fontsize=7.5, color="#666")
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Saliency z of a patch (standardised evidence)")
    ax.set_ylabel("Share of learned weight kept")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#EBEBEB", lw=0.6)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    save(fig, "fig_gate_curves")


def main():
    heads = load_stats()["heads"]
    hard = (heads["clean/gated"]["scalars"]["gamma"], heads["clean/gated"]["scalars"]["delta"])
    soft = heads["clean/soft_gated"]["scalars"]
    gate_curves(hard, (soft["gamma"], soft["delta"], soft["floor"]))


if __name__ == "__main__":
    main()
