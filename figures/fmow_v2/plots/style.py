"""Shared plot style, colours, data paths and save helper for the paper figures."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"  # summary JSON / NPZ inputs (see data/README.md)
FIGURES = Path(os.environ.get("FIG_DIR", HERE.parent / "figures"))
EXAMPLES = Path(os.environ.get("EXAMPLE_DIR", HERE.parent / "saliency_examples"))

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.edgecolor": "#555",
        "axes.linewidth": 0.8,
        "grid.color": "#ebebeb",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)

CLEAN, CLOUD, NEUT, LIGHT = "#4C72B0", "#DD8452", "#8C8C8C", "#C8C8C8"
HC = {
    "Learned attention": "#8172B3",
    "Hard gate": "#C44E52",
    "Hard prior": "#937860",
    "Soft gate": "#64B5CD",
    "Guided": "#55A868",
    "Input": "#4C72B0",
}


def load_stats():
    """All v2 summary numbers (built by stats.py from the run outputs)."""
    return json.loads((DATA / "v2_stats.json").read_text())


def letter(ax, label):
    """Prefix a panel title with (a), (b), ..."""
    ax.set_title(f"({label})  " + ax.get_title(loc="left"), loc="left")


def save(fig, name, folder=None):
    folder = Path(folder or FIGURES)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
