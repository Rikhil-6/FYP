"""Create a compact two-panel view of routing gains over the historical ViT."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "transparent_router" / "20261006-drift-v2" / "drift_v2_2014_metrics.csv"
OUTPUT = ROOT / "reports" / "transparent_router" / "20261006-drift-v2" / "drift_v2_router_tradeoff.png"

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
    data = pd.read_csv(INPUT)
    data["condition"] = data["scenario"].map(LABELS)
    data = data.sort_values("saliency_routed_delta_pp", ascending=False).reset_index(drop=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
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

    left.barh(
        y,
        data["saliency_routed_delta_pp"],
        height=height,
        color=routed_color,
        edgecolor="white",
        linewidth=0.6,
    )
    left.axvline(0, color="#444444", linewidth=0.9)
    left.set_yticks(y, data["condition"])
    left.invert_yaxis()
    left.set_xlim(-0.12, 4.25)
    left.set_xlabel("Routed macro F1 gain over historical ViT (percentage points)")
    left.set_title("A. Selective routing versus the frozen historical ViT", loc="left", weight="bold")

    effects = data["auxiliary_routed_effect_pp"].to_numpy()
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
