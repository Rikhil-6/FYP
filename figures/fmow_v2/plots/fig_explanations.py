"""Exact explanation panels: old model's 20x20 evidence vs the updated hard gate's attention x evidence (Section 6.6).

Writes fig_explain_goodbad, fig_explain_update and one PNG per example image.
Example::  python fig_explanations.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom

from style import DATA, EXAMPLES, FIGURES, save

plt.rcdefaults()  # image panels: plain style, no seaborn grid
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 7.4,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)
d = np.load(DATA / "explain_examples.npz")
d20 = np.load(DATA / "explain20.npz")
names = [str(n).replace("_", " ") for n in d["names"]]
EARLIER = [
    "both correct",
    "both correct",
    "both correct",
    "helpful switch",
    "helpful switch",
    "harmful switch",
    "real cloud",
    "real cloud",
]


def wrap(t, n=16):
    return t if len(t) <= n else t[: n - 1] + "…"


def support(m):
    """Positive evidence as a share of total support, with cells outside the smallest set holding half of it faded to zero."""
    pos = np.clip(m, 0, None)
    pos = pos / max(pos.sum(), 1e-09)
    srt = np.sort(pos)[::-1]
    thr = srt[min(np.searchsorted(srt.cumsum(), 0.5), len(srt) - 1)]
    return (np.where(pos >= thr, pos, 0.0), int((pos >= thr).sum()), len(pos))


def smooth(m):
    G = int(round(np.sqrt(m.size)))
    return np.clip(zoom(m.reshape(G, G), 280 / G, order=1), 0, None)


def base(ax, img, box, title):
    ax.imshow(img.mean(-1) / 255, cmap="gray", vmin=0, vmax=1.15)
    cx, cy, w, h = box
    ax.add_patch(
        Rectangle(((cx - w / 2) * 280, (cy - h / 2) * 280), w * 280, h * 280, fill=False, ec="#111", lw=1.0, ls="--")
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left")
    [s.set_visible(False) for s in ax.spines.values()]


def heat(ax, m, cmap, vmax):
    a = smooth(m)
    ax.imshow(np.ma.masked_less_equal(a, vmax * 0.08), cmap=cmap, vmin=0, vmax=vmax, alpha=0.8)


def row(axs, k, label):
    img, box = (d["images"][k], d["boxes"][k])
    po, pn = (d["old"][k], d20["gated_pred"][k])
    so, ko, No = support(d["old_ev20"][k])
    sn, kn, Nn = support(d20["gated_c20"][k])
    fo = np.clip(d["old_ev20"][k], 0, None)
    fo = fo / fo.sum()
    fn = np.clip(d20["gated_c20"][k], 0, None)
    fn = fn / fn.sum()
    ax0 = axs[0]
    ax0.imshow(img)
    cx, cy, w, h = box
    ax0.add_patch(
        Rectangle(((cx - w / 2) * 280, (cy - h / 2) * 280), w * 280, h * 280, fill=False, ec="white", lw=1.0, ls="--")
    )
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title(f"{label}\ntrue: {wrap(names[d['labels'][k]], 15)}", loc="left")
    [s.set_visible(False) for s in ax0.spines.values()]
    base(axs[1], img, box, f"Old: {wrap(names[po], 13)}\nhalf of support: {ko}/{No}")
    heat(axs[1], fo, "Reds", smooth(fo).max())
    base(axs[2], img, box, f"New: {wrap(names[pn], 13)}\nhalf of support: {kn}/{Nn}")
    heat(axs[2], fn, "Oranges", smooth(fn).max())
    full_o, full_n = (fo, fn)
    diff = zoom((full_n - full_o).reshape(20, 20), 14, order=1)
    lim = np.abs(diff).max()
    axs[3].imshow(img.mean(-1) / 255, cmap="gray", vmin=0, vmax=1.15)
    axs[3].imshow(diff, cmap="PuOr_r", vmin=-lim, vmax=lim, alpha=0.7)
    axs[3].add_patch(
        Rectangle(((cx - w / 2) * 280, (cy - h / 2) * 280), w * 280, h * 280, fill=False, ec="#111", lw=1.0, ls="--")
    )
    axs[3].set_xticks([])
    axs[3].set_yticks([])
    axs[3].set_title("Change after update\norange = new relies more", loc="left")
    [s.set_visible(False) for s in axs[3].spines.values()]


def figure(ks, labels, fname, folder=FIGURES):
    fig, ax = plt.subplots(len(ks), 4, figsize=(7.0, 2.0 * len(ks)), gridspec_kw={"wspace": 0.08, "hspace": 0.36})
    ax = np.atleast_2d(ax)
    for i, k in enumerate(ks):
        row(ax[i], k, labels[i])
    if len(ks) > 1:
        fig.text(
            0.5,
            0.06 if len(ks) > 3 else 0.075,
            "Columns 2–3: exact positive contributions of each 20 × 20 patch to each model's score for its own prediction (darker = larger share); titles give the number of patches supplying half of it.\nColumn 4: share of support gained (orange) or lost (purple) by each region after the update. Dashed box = labelled object.",
            fontsize=6.6,
            color="#444",
            ha="center",
            va="top",
        )
    save(fig, fname, folder)


def main():
    groups = list(d["group"])
    good = [i for i, x in enumerate(groups) if x == "good"]
    bad = [i for i, x in enumerate(groups) if x == "bad"]
    earlier = [i for i, x in enumerate(groups) if x == "earlier"]
    figure(good[:3] + bad[:3], ["Object-aligned"] * 3 + ["Diffuse"] * 3, "fig_explain_goodbad")
    figure(
        [earlier[1], earlier[3], earlier[4], earlier[5]],
        ["Correct, both", "Helpful switch", "Helpful switch", "Harmful switch"],
        "fig_explain_update",
    )
    for i, group in enumerate(groups):
        label = EARLIER[i - 12] if group == "earlier" else ("object-aligned" if group == "good" else "diffuse")
        cls = names[d["labels"][i]].replace(" ", "_")
        figure([i], [label.capitalize()], f"{i + 1:02d}_{label.replace(' ', '_')}_{cls}", EXAMPLES)


if __name__ == "__main__":
    main()
