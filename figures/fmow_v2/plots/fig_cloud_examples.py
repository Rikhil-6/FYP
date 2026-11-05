"""Cloud versions of Figures 10 and 12: head attention weights and old/new exact explanations on test images carrying the training cloud overlay.

Input: cloud_examples.npz written by fyp_drift.evaluation.fmow_v2.cloud_examples. Writes fig_maps_cloud and fig_explain_cloud.
Example::  python fig_cloud_examples.py cloud_examples.npz outdir
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom

plt.rcdefaults()
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 6.6, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
d = np.load(sys.argv[1], allow_pickle=True)
OUT = sys.argv[2]
names = [str(n).replace("_", " ") for n in d["names"]]
BOX = "#E0662B"


def wrap(t, n=15):
    return t if len(t) <= n else t[: n - 1] + "…"


def box(ax, b, color):
    cx, cy, w, h = b
    ax.add_patch(Rectangle(((cx - w / 2) * 280, (cy - h / 2) * 280), w * 280, h * 280, fill=False, ec=color, lw=1.2, ls="--"))
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]


def smooth(m):
    G = int(round(np.sqrt(m.size)))
    return np.clip(zoom(m.reshape(G, G), 280 / G, order=1), 0, None)


def maps():
    cols = [("Image", None), ("Learned\nattention", "attention"), ("Hard gate", "gated"), ("Hard gate,\nshuffled map", "gated_shuffled")]
    fig, axes = plt.subplots(6, 4, figsize=(5.8, 9.0), gridspec_kw={"wspace": 0.05, "hspace": 0.2})
    cc = (np.arange(10) + 0.5) / 10
    X, Y = np.meshgrid(cc, cc)
    for r in range(3):
        b = d["boxes"][r]
        inb = ((np.abs(X - b[0]) <= b[2] / 2) & (np.abs(Y - b[1]) <= b[3] / 2)).ravel()
        for v, (tag, img) in enumerate((("clean", d["images"][r]), ("cloud", d["images_cloud"][r]))):
            row = 2 * r + v
            for c, (title, key) in enumerate(cols):
                ax = axes[row, c]
                if key is None:
                    ax.imshow(img)
                else:
                    ax.imshow(img.mean(-1), cmap="gray", vmin=0, vmax=300)
                    w = d[f"{key}_w_{tag}"][r].astype(np.float32); w = w / w.sum()
                    a = np.clip(zoom(w.reshape(10, 10), 28, order=1), 0, None)
                    ax.imshow(np.ma.masked_less(a, a.max() * 0.12), cmap="Blues", vmin=0, vmax=a.max(), alpha=0.85)
                    ax.text(6, 272, f"{100 * w[inb].sum():.0f}% on object", fontsize=6.6, color="white", va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="#222", ec="none", alpha=0.7))
                box(ax, b, "white" if key is None else BOX)
                if row == 0: ax.set_title(title, fontsize=8, loc="center", fontweight="bold")
                if c == 0: ax.set_ylabel(f"{names[d['labels'][r]]}\n({'clean' if tag == 'clean' else 'with cloud'})", fontsize=7.6)
    fig.savefig(f"{OUT}/fig_maps_cloud.png"); plt.close(fig)


def explain():
    labels = ["Correct, both", "Wrong, both", "Helpful switch", "Correct, both"]
    fig, ax = plt.subplots(4, 4, figsize=(7.0, 8.0), gridspec_kw={"wspace": 0.08, "hspace": 0.36})
    for i in range(4):
        k = 3 + i; img = d["images_cloud"][k]; b = d["boxes"][k]
        po, pn = d["old_pred_cloud"][k], d["gated_pred_cloud"][k]
        fo = np.clip(d["old_ev20_cloud"][k], 0, None); fo /= fo.sum()
        fn = np.clip(d["gated_c20_cloud"][k], 0, None); fn /= fn.sum()
        def half(m):
            s = np.sort(m)[::-1]; return int(np.searchsorted(s.cumsum(), 0.5)) + 1
        ax[i, 0].imshow(img); box(ax[i, 0], b, "white")
        ax[i, 0].set_title(f"{labels[i]}\ntrue: {wrap(names[d['labels'][k]])}", loc="left")
        for c, (m, cm, nm, p) in enumerate(((fo, "Reds", "Old", po), (fn, "Oranges", "New", pn)), start=1):
            a = ax[i, c]; a.imshow(img.mean(-1) / 255, cmap="gray", vmin=0, vmax=1.15)
            s = smooth(m); a.imshow(np.ma.masked_less_equal(s, s.max() * 0.08), cmap=cm, vmin=0, vmax=s.max(), alpha=0.8)
            box(a, b, "#111"); a.set_title(f"{nm}: {wrap(names[p], 13)}\nhalf: {half(m)}/400 patches", loc="left")
        diff = zoom((fn - fo).reshape(20, 20), 14, order=1); lim = np.abs(diff).max()
        ax[i, 3].imshow(img.mean(-1) / 255, cmap="gray", vmin=0, vmax=1.15); ax[i, 3].imshow(diff, cmap="PuOr_r", vmin=-lim, vmax=lim, alpha=0.7)
        box(ax[i, 3], b, "#111"); ax[i, 3].set_title("Change after update\norange = new relies more", loc="left")
    fig.savefig(f"{OUT}/fig_explain_cloud.png"); plt.close(fig)


maps(); explain()
