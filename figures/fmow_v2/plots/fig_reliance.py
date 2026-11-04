"""Reliance and audit figures: moved-object test, weight maps, saliency in training, router, gate curves (Sections 3, 6, 7).

Example::  python fig_reliance.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from style import CLEAN, CLOUD, DATA, HC, LIGHT, NEUT, letter, load_stats, save

D = load_stats()


def fig_shift():
    SH = D["shift"]
    arms = [
        ("attention", "Learned attention"),
        ("gated", "Hard gate"),
        ("gated_shuffled", "Hard gate, shuffled"),
        ("prior", "Hard prior"),
        ("prior_shuffled", "Hard prior, shuffled"),
        ("soft_gated", "Soft gate"),
        ("soft_gated_shuffled", "Soft gate, shuffled"),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.35})
    y = np.arange(len(arms))
    for k, (arm, lab) in enumerate(arms):
        s = SH[f"clean/{arm}"]
        c0, c1 = (s["box_mass_minus_centre_prior_c"], s["box_mass_minus_centre_prior_s"])
        col = NEUT if "shuffled" in arm else HC.get(lab, "#333")
        a.plot([c0, c1], [k, k], color="#ccc", lw=2, zorder=1)
        a.scatter([c0], [k], marker="o", s=42, color=col, zorder=3, edgecolor="white")
        a.scatter([c1], [k], marker="s", s=38, facecolor="white", edgecolor=col, lw=1.4, zorder=3)
    a.axvline(0, color="#333", lw=0.8)
    a.set_yticks(y, [label for _, label in arms])
    a.set_ylim(len(arms) - 0.4, -0.6)
    a.grid(axis="y", visible=False)
    a.set_xlabel("Weight on object minus centred-prior weight")
    a.set_title("Object weight beyond centre bias", pad=22)
    letter(a, "a")
    a.legend(
        [
            Line2D([], [], marker="o", color="#555", ls=""),
            Line2D([], [], marker="s", markerfacecolor="white", color="#555", ls=""),
        ],
        ["Centred object", "Moved object"],
        loc="lower right",
        fontsize=7.5,
    )
    names = ["Sal.\nmap", "Shuf.\nmap", "Learned\nattn.", "Hard\ngate", "Centre\nonly"]
    cen = [
        SH["clean/gated"]["salmap_point_c"],
        SH["clean/gated_shuffled"]["salmap_point_c"],
        SH["clean/attention"]["pointing_game_c"],
        SH["clean/gated"]["pointing_game_c"],
        SH["clean/attention"]["centre_prior_pointing_c"],
    ]
    mov = [
        SH["clean/gated"]["salmap_point_s"],
        SH["clean/gated_shuffled"]["salmap_point_s"],
        SH["clean/attention"]["pointing_game_s"],
        SH["clean/gated"]["pointing_game_s"],
        SH["clean/attention"]["centre_prior_pointing_s"],
    ]
    x = np.arange(5)
    b.bar(x - 0.19, cen, width=0.38, color=LIGHT, label="Centred")
    b.bar(x + 0.19, mov, width=0.38, color=CLEAN, label="Moved")
    b.set_xticks(x, names, fontsize=7.4)
    b.set_ylim(0, 1.05)
    b.set_ylabel("Pointing game")
    b.set_title("Top patch on the object", pad=22)
    b.grid(axis="x", visible=False)
    b.legend(loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.1))
    letter(b, "b")
    save(fig, "fig_shift")


PICK = [3, 5, 1]


def fig_maps():
    from scipy.ndimage import zoom

    z = np.load(DATA / "examples.npz", allow_pickle=True)
    zs = np.load(DATA / "examples_soft.npz")
    pick = PICK

    def W(key, v, i):
        return zs[f"soft_gated_{v}"][i] if key == "soft_gated" else z[f"{key}_{v}"][i]

    cols = [
        ("Image", None),
        ("Learned\nattention", "attention"),
        ("Hard gate", "gated"),
        ("Soft gate", "soft_gated"),
        ("Hard gate,\nshuffled map", "gated_shuffled"),
    ]
    fig, axes = plt.subplots(
        2 * len(pick), 5, figsize=(7.0, 1.5 * 2 * len(pick)), gridspec_kw={"wspace": 0.05, "hspace": 0.22}
    )
    cc = (np.arange(10) + 0.5) / 10
    X, Y = np.meshgrid(cc, cc)
    for r, i in enumerate(pick):
        for v_i, v in enumerate(("centred", "shifted")):
            row = 2 * r + v_i
            img = z[v][i]
            box = z["boxes" if v == "centred" else "sboxes"][i]
            cx, cy, bw, bh = box
            inb = ((np.abs(X - cx) <= bw / 2) & (np.abs(Y - cy) <= bh / 2)).ravel()
            for c, (title, key) in enumerate(cols):
                ax = axes[row, c]
                ax.set_xticks([])
                ax.set_yticks([])
                [sp.set_visible(False) for sp in ax.spines.values()]
                if key is None:
                    ax.imshow(img)
                else:
                    ax.imshow(img.mean(-1), cmap="gray", vmin=0, vmax=300)
                    w = W(key, v, i).astype(np.float32)
                    w = w / w.sum()
                    a = np.clip(zoom(w.reshape(10, 10), 28, order=1), 0, None)
                    ax.imshow(np.ma.masked_less(a, a.max() * 0.12), cmap="Blues", vmin=0, vmax=a.max(), alpha=0.85)
                    ax.text(
                        6,
                        272,
                        f"{100 * w[inb].sum():.0f}% on object",
                        fontsize=6.6,
                        color="white",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#222", ec="none", alpha=0.7),
                    )
                ax.add_patch(
                    Rectangle(
                        ((cx - bw / 2) * 280, (cy - bh / 2) * 280),
                        bw * 280,
                        bh * 280,
                        fill=False,
                        ec="white" if key is None else CLOUD,
                        lw=1.2,
                        ls="--",
                    )
                )
                if row == 0:
                    ax.set_title(title, fontsize=8, loc="center", fontweight="bold")
                if c == 0:
                    ax.set_ylabel(
                        f"{str(z['classes'][i]).replace('_', ' ')}\n({('centred' if v == 'centred' else 'moved')})",
                        fontsize=7.6,
                    )
    save(fig, "fig_maps")


def fig_learning():
    SL = D["sal_learning"]
    pairs = [
        ("sgt", "sgt_random", "Saliency-guided training\nvs same with random mask"),
        ("focus_aug", "random_aug", "Drop least-salient patches\nvs drop random patches"),
        ("context_aug", "random_aug", "Drop most-salient patches\nvs drop random patches"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), gridspec_kw={"wspace": 0.2}, sharey=True)
    y = np.arange(len(pairs))

    def bars(ax, vals, fmt, title, xlab, lim, good):
        cols = ["#55A868" if (v > 0) == good else "#C44E52" for v in vals]
        ax.barh(y, vals, color=cols, height=0.55)
        ax.axvline(0, color="#333", lw=0.9)
        for k, v in enumerate(vals):
            ax.text(
                v + (0.04 if v >= 0 else -0.04) * (lim[1] - lim[0]),
                k,
                fmt.format(v),
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=7.8,
            )
        ax.set_xlim(*lim)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlab, fontsize=7.6)
        ax.grid(axis="y", visible=False)

    acc = [
        np.mean([SL[f"{c}/{a}"]["f1"][0] - SL[f"{c}/{b}"]["f1"][0] for c in ("clean", "cloud")]) for a, b, _ in pairs
    ]
    foc = [SL[f"clean/{a}"]["focus"] - SL[f"clean/{b}"]["focus"] for a, b, _ in pairs]
    dep = [SL[f"clean/{a}"]["del_top25"] - SL[f"clean/{b}"]["del_top25"] for a, b, _ in pairs]
    bars(axes[0], acc, "{:+.1f}", "Accuracy", "Macro-F1 change (points)", (-5, 2), True)
    bars(axes[1], foc, "{:+.1f}×", "Focus on the object", "Change in object weight ÷ area", (-0.5, 1.6), True)
    bars(
        axes[2],
        dep,
        "{:+.1f}",
        "Reliance on top patches",
        "Change in accuracy lost when\ntop 25% of patches removed",
        (-13, 4),
        False,
    )
    axes[0].set_yticks(y, [p[2] for p in pairs], fontsize=7.6)
    axes[0].set_ylim(len(pairs) - 0.5, -0.5)
    for a, label in zip(axes, "abc", strict=True):
        letter(a, label)
    fig.text(
        0.5,
        -0.2,
        "Each bar is a saliency method minus its matched control (mean of clean and cloudy test for accuracy; clean images otherwise). Green = desirable direction, red = undesirable.",
        ha="center",
        fontsize=7,
        color="#444",
    )
    save(fig, "fig_learning")


def fig_router():
    R = D["router"]
    settings = [
        ("clean/gated", "Clean, hard gate", CLEAN, "o"),
        ("clean/input", "Clean, input head", CLEAN, "s"),
        ("cloud/gated", "Cloudy, hard gate", CLOUD, "o"),
        ("cloud/input", "Cloudy, input head", CLOUD, "s"),
    ]
    pols = [
        ("confidence_only", "Switch on confidence\n(probability, entropy)"),
        ("saliency_only", "Switch on saliency\nsignals only"),
        ("combined", "Switch on all four\nsignals"),
    ]
    fig = plt.figure(figsize=(7.6, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.55)
    a = fig.add_subplot(gs[0])
    b = fig.add_subplot(gs[1])
    for pi, (p, _label) in enumerate(pols):
        for si, (s_, sl, c, mk) in enumerate(settings):
            m, sd = R[f"{s_}/{p}"]["diff"][:2]
            yy = pi + (si - 1.5) * 0.15
            a.errorbar(
                m, yy, xerr=sd, fmt=mk, color=c, ms=5.5, capsize=2, lw=0.9, mec="white", label=sl if pi == 0 else None
            )
    a.axvline(0, color="#333", lw=0.9)
    a.axvspan(-3, 0, color="#C44E52", alpha=0.05, lw=0)
    a.text(0.08, -0.38, "router better →", fontsize=7, color="#555")
    a.text(-0.08, -0.38, "← always switching better", fontsize=7, color="#555", ha="right")
    a.set_yticks(range(3), [p[1] for p in pols], fontsize=7.6)
    a.set_ylim(2.5, -0.6)
    a.set_xlim(-2.5, 1.0)
    a.grid(axis="y", visible=False)
    a.set_xlabel("Macro-F1 of router minus always using the\nnew model (points, mean ± sd over 3 seeds)", fontsize=7.6)
    a.set_title("Does the router beat always switching?")
    letter(a, "a")
    a.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.45, -0.3), frameon=False, ncol=2)
    groups = [
        ("helpful_route", "Helpful switch\n(new right,\nold wrong)"),
        ("harmful_route", "Harmful switch\n(new wrong,\nold right)"),
    ]
    for gi, (g, _label) in enumerate(groups):
        for mi, (m, ml, c) in enumerate((("updated_map", "new\nmodel", CLEAN), ("historical_map", "old\nmodel", NEUT))):
            vals = [R[f"{s}/combined"]["dm"][f"{g}/{m}"] for s, *_ in settings]
            xx = gi + (mi - 0.5) * 0.4
            mv = np.mean(vals)
            b.bar(xx, mv, width=0.36, color=c)
            b.scatter([xx] * len(vals), vals, color="#222", s=8, zorder=3)
            b.text(xx, max(vals) + 0.03 if mv > 0 else 0.03, f"{mv:+.2f}\n{ml}", ha="center", va="bottom", fontsize=7)
    b.axhline(0, color="#333", lw=0.9)
    b.set_xticks([0, 1], [g[1] for g in groups], fontsize=7.6)
    b.set_ylim(-0.2, 0.82)
    b.set_xlim(-0.55, 1.55)
    b.set_ylabel("Deletion margin for the true class", fontsize=7.6)
    b.grid(axis="x", visible=False)
    b.set_title("Whose map supports the true class?")
    letter(b, "b")
    save(fig, "fig_router")


def main():
    for make in (fig_shift, fig_maps, fig_learning, fig_router):
        make()


if __name__ == "__main__":
    main()
