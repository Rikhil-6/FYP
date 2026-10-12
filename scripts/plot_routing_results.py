"""Create a compact two-panel view of routing gains over the historical ViT."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "drift" / "metrics_2014.csv"
OUTPUT = ROOT / "paper" / "figures" / "routing_results.png"

LABELS = {
    "clean": "Clean",
    "illumination": "Illumination",
    "white_balance": "White balance",
    "sensor_noise": "Sensor noise",
    "defocus_blur": "Defocus blur",
    "motion_blur": "Motion blur",
    "jpeg_compression": "JPEG compression",
    "resolution_loss": "Resolution loss",
    "viewpoint_shift": "Viewpoint shift",
    "occlusion": "Occlusion",
    "realistic_mix": "Realistic mixture",
}


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))
    for row in data:
        row["condition"] = LABELS[row["scenario"]]
        row["saliency_routed_delta_pp"] = float(row["saliency_routed_delta_pp"])
        row["auxiliary_routed_effect_pp"] = float(row["auxiliary_routed_effect_pp"])
    data.sort(key=lambda row: row["saliency_routed_delta_pp"], reverse=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(15, 913 / 120),
        dpi=120,
        gridspec_kw={"width_ratios": [2.25, 1], "wspace": 0.12},
        sharey=True,
    )

    y = np.arange(len(data))
    height = 0.48
    routed_color = "#2171b5"

    routed_gain = np.array([row["saliency_routed_delta_pp"] for row in data])
    left.barh(
        y,
        routed_gain,
        height=height,
        color=routed_color,
        edgecolor="white",
        linewidth=0.6,
    )
    left.axvline(0, color="#444444", linewidth=0.9)
    left.set_yticks(y, [row["condition"] for row in data])
    left.invert_yaxis()
    left.set_xlim(-0.12, 4.25)
    left.set_xlabel("Routed macro F1 gain over historical ViT (percentage points)")
    left.set_title("A. Selective routing versus the frozen historical ViT", loc="left", weight="bold")

    effects = np.array([row["auxiliary_routed_effect_pp"] for row in data])
    effect_colors = np.where(effects >= 0, "#2171b5", "#b26a5b")
    right.barh(y, effects, height=0.48, color=effect_colors, edgecolor="white", linewidth=0.6)
    right.axvline(0, color="#444444", linewidth=0.9)
    right.set_xlim(-0.25, 1.0)
    right.tick_params(axis="y", left=False, labelleft=False)
    right.set_xlabel("Saliency router minus no-aux router (points)")
    right.set_title("B. Added effect of saliency losses", loc="left", weight="bold")
    right.legend(
        handles=[
            Patch(facecolor="#2171b5", label="Saliency router better"),
            Patch(facecolor="#b26a5b", label="No-aux router better"),
        ],
        frameon=False,
        loc="lower right",
    )

    for axis in (left, right):
        axis.grid(axis="y", visible=False)
        axis.grid(axis="x", color="#d9d9d9", linewidth=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.set_axisbelow(True)

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.16, right=0.985, top=0.93, bottom=0.13)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, facecolor="white")
    plt.close(fig)

    with Image.open(OUTPUT) as image:
        if image.size != (1800, 913):
            image.resize((1800, 913), Image.Resampling.LANCZOS).save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
