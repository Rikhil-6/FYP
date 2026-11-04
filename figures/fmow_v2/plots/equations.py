"""Render the paper's equations as 300-dpi PNGs (the Word document embeds them as images).

Example::  python equations.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "equations"
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})

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
    "grounding": r"$\mathrm{focus} = \dfrac{\sum_{i \in B} a_i}{|B| / N}, \qquad \mathrm{pointing} = \mathbf{1}\left[\arg\max_i a_i \in B\right], \qquad \mathrm{IoU} = \dfrac{|T_k \cap B|}{|T_k \cup B|}$",
    "ece": r"$\mathrm{ECE} = \sum_{b=1}^{10} \dfrac{|S_b|}{n}\, \left| \mathrm{acc}(S_b) - \mathrm{conf}(S_b) \right|$",
    "js": r"$\mathrm{JS}(P, Q) = \frac{1}{2}\mathrm{KL}(P \,\|\, M) + \frac{1}{2}\mathrm{KL}(Q \,\|\, M), \qquad M = \frac{1}{2}(P + Q)$",
    "router": r"$\Pr(\mathrm{update} \mid x) = \sigma\left(b + w^{\top} u\right), \qquad \mathrm{use}\ p_{\mathrm{new}} \Leftrightarrow \hat{y}_{\mathrm{old}} \neq \hat{y}_{\mathrm{new}}\ \wedge\ u_1 \geq \tau_{\mathrm{JS}}\ \wedge\ \Pr(\mathrm{update} \mid x) \geq \tau_p$",
    "kept": r"$\mathrm{kept} = \dfrac{F_{\mathrm{routed}} - F_{\mathrm{old}}}{F_{\mathrm{new}} - F_{\mathrm{old}}}$",
    "margin": r"$m = \left[p_y(x) - p_y(x \setminus T_{20})\right] - \left[p_y(x) - p_y(x \setminus B_{20})\right]$",
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, tex in EQ.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=13)
        fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
