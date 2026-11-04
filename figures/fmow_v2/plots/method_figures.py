"""Method figures: the learned gate response curves and the paper's equations as 300-dpi PNGs.

The system schematics are TikZ sources in ../tikz.
Example::  python method_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import load_stats, save

TRAIN_E, SAL_E = "#C46A2F", "#4C72B0"
EQUATIONS = Path(__file__).resolve().parent.parent / "equations"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


EQ = {
    "tokens": r"$h_i = f(x)_i \in \mathbb{R}^{1024}, \qquad i = 1, \dots, N, \quad N = 100$",
    "old": r"$p_{\mathrm{old}}(y \mid x) = \mathrm{softmax}\left(W \bar{h} + b\right), \qquad \bar{h} = \frac{1}{N}\sum_{i=1}^{N} h_i, \qquad \hat{y} = \arg\max_y\, p_{\mathrm{old}}(y \mid x)$",
    "saliency": r"$z_i = \dfrac{e_i - \mathrm{mean}_j(e_j)}{\mathrm{std}_j(e_j)}, \qquad e_i = w_{\hat{y}}^{\top} h_i$",
    "shuffle": r"$\tilde{z}_i = z_{\pi(i)}, \qquad \pi \sim \mathrm{Uniform}(S_N)$",
    "attention": r"$a_i = \mathrm{softmax}_i(s_i), \qquad s_i = g(h_i)$",
    "prior": r"$a_i = \mathrm{softmax}_i(\beta z_i + s_i), \qquad s \equiv 0,\ \beta = 1 \ \mathrm{at\ initialisation}$",
    "gate": r"$a_i = \dfrac{\mathrm{softmax}_i(s_i)\, \sigma(\gamma z_i + \delta)}{\sum_j \mathrm{softmax}_j(s_j)\, \sigma(\gamma z_j + \delta)}$",
    "softgate": r"$a_i \propto \mathrm{softmax}_i(s_i)\, \left[\rho + (1-\rho)\, \sigma(\gamma z_i + \delta)\right], \qquad \rho = \sigma(r) \in (0, 1)$",
    "input": r"$a_i = \mathrm{softmax}_i\left(g([h_i\, ;\, z_i])\right)$",
    "guided": r"$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda\, \mathrm{KL}\left(\mathrm{softmax}(z)\, \|\, a\right)$",
    "mix": r"$a_i = (1-\lambda)\, \mathrm{softmax}_i(s_i) + \lambda\, \mathrm{softmax}_i(\beta z_i), \qquad \lambda = \sigma(l)$",
    "new": r"$p_{\mathrm{new}}(y \mid x) = \mathrm{softmax}\left(W' \sum_{i=1}^{N} a_i h_i + b'\right)$",
    "lora": r"$W = W_0 + \frac{\alpha}{r} B A, \qquad A \in \mathbb{R}^{r \times d_{\mathrm{in}}},\ B \in \mathbb{R}^{d_{\mathrm{out}} \times r}, \qquad |\theta| = 6 \times 18{,}432\, r$",
    "anchor": r"$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda \sum_{i=1}^{N} w_i \, \left\| h_i^{\mathrm{LoRA}} - h_i^{\mathrm{frozen}} \right\|_2^2, \qquad \sum_i w_i = 1$",
    "mmd": r"$\mathrm{MMD}^2 = \mathbb{E}[k(h, h')] + \mathbb{E}[k(g, g')] - 2\,\mathbb{E}[k(h, g)], \qquad k(u, v) = \exp\left(-\frac{\|u - v\|^2}{2\ell^2}\right)$",
    "gap": r"$\Delta_{\mathrm{harm}} = \mathrm{Acc}_{\mathrm{val},\ 2010–13} - \mathrm{Acc}_{\mathrm{test},\ 2016–17}$",
    "deletion": r"$\Delta_{\mathrm{top}}(f) = \mathrm{Acc}(a) - \mathrm{Acc}\left(\frac{a \odot m_{\mathrm{top}}(f)}{\| a \odot m_{\mathrm{top}}(f) \|_1}\right), \qquad \mathrm{flip} = \Pr\left[\hat{y}(a) \neq \hat{y}(\mathbf{1}/N)\right]$",
    "grounding": r"$\mathrm{focus} = \dfrac{\sum_{i \in B} a_i}{|B| / N}, \qquad \mathrm{pointing} = \mathbf{1}\left[\arg\max_i a_i \in B\right]$",
    "ece": r"$\mathrm{ECE} = \sum_{b=1}^{10} \dfrac{|S_b|}{n}\, \left| \mathrm{acc}(S_b) - \mathrm{conf}(S_b) \right|$",
    "js": r"$\mathrm{JS}(P, Q) = \frac{1}{2}\mathrm{KL}(P \,\|\, M) + \frac{1}{2}\mathrm{KL}(Q \,\|\, M), \qquad M = \frac{1}{2}(P + Q)$",
    "router": r"$\Pr(\mathrm{update} \mid x) = \sigma\left(b + w^{\top} u\right), \qquad \mathrm{use}\ p_{\mathrm{new}} \Leftrightarrow \hat{y}_{\mathrm{old}} \neq \hat{y}_{\mathrm{new}}\ \wedge\ u_1 \geq \tau_{\mathrm{JS}}\ \wedge\ \Pr(\mathrm{update} \mid x) \geq \tau_p$",
    "kept": r"$\mathrm{kept} = \dfrac{F_{\mathrm{routed}} - F_{\mathrm{old}}}{F_{\mathrm{new}} - F_{\mathrm{old}}}$",
    "margin": r"$m = \left[p_y(x) - p_y(x \setminus T_{20})\right] - \left[p_y(x) - p_y(x \setminus B_{20})\right]$",
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gate_curves(hard, soft):
    """Share of learned weight kept vs saliency, for the learned hard gate (gamma, delta) and soft gate (gamma, delta, rho)."""
    z = np.linspace(-3, 3, 300)
    g_hard = sigmoid(hard[0] * z + hard[1])
    g_soft = soft[2] + (1 - soft[2]) * sigmoid(soft[0] * z + soft[1])
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.axvspan(-3, -1, color="#F3F3F3", lw=0)
    ax.plot(
        z,
        np.ones_like(z),
        color="#8C8C8C",
        lw=1.4,
        ls=":",
        label="Learned attention (no gate)",
    )
    ax.plot(
        z,
        g_soft,
        color=SAL_E,
        lw=1.8,
        ls="--",
        label=f"Soft gate (floor ρ = {soft[2]:.2f})",
    )
    ax.plot(
        z,
        g_hard,
        color=TRAIN_E,
        lw=1.8,
        label=f"Hard gate (γ = {hard[0]:.2f}, δ = {hard[1]:.2f})",
    )
    ax.text(
        -2.0, 0.08, "low-evidence\npatches", ha="center", fontsize=7.5, color="#666"
    )
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Saliency z of a patch (standardised evidence)")
    ax.set_ylabel("Share of learned weight kept")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#EBEBEB", lw=0.6)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    save(fig, "fig_gate_curves")


def equations():
    """Render each equation of the paper to its own PNG (the Word document embeds them as images)."""
    EQUATIONS.mkdir(exist_ok=True)
    with plt.rc_context({"mathtext.fontset": "cm", "font.family": "serif"}):
        for name, tex in EQ.items():
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, tex, fontsize=13)
            fig.savefig(
                EQUATIONS / f"{name}.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.04,
                facecolor="white",
            )
            plt.close(fig)


def main():
    equations()
    heads = load_stats()["heads"]
    hard = (
        heads["clean/gated"]["scalars"]["gamma"],
        heads["clean/gated"]["scalars"]["delta"],
    )
    soft = heads["clean/soft_gated"]["scalars"]
    gate_curves(hard, (soft["gamma"], soft["delta"], soft["floor"]))


if __name__ == "__main__":
    main()
