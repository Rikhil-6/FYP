from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
COLOURS = {
    "Historical ViT": "#616161",
    "Updated XAI ViT": "#56B4E9",
    "Transparent router": "#0072B2",
}


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def plot_temporal_performance():
    data = pd.read_csv(ROOT / "results" / "temporal_performance.csv")
    columns = {
        "historical_xai": "Historical ViT",
        "updated_xai": "Updated XAI ViT",
        "routed": "Transparent router",
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharex=True)
    for axis, condition in zip(axes, ("Clean", "Combined drift")):
        subset = data[data["condition"] == condition]
        for column, label in columns.items():
            axis.plot(
                subset["year"],
                subset[column],
                marker="o",
                linewidth=2,
                markersize=5,
                label=label,
                color=COLOURS[label],
            )
        axis.set_title(condition, fontweight="bold")
        axis.set_xticks([2012, 2013, 2014])
        axis.set_xlabel("Evaluation year")
        axis.set_ylabel("Macro F1 (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "temporal_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def readable_shift_name(target):
    name = target.replace("_severity_3", "").replace("_", " ")
    label = name.replace("jpeg", "JPEG").title().replace("Jpeg", "JPEG")
    return label.replace("Realistic Mix", "Realistic Mixture")


def plot_drift_verification():
    data = pd.read_csv(ROOT / "results" / "drift_verification" / "clear100_300.csv")
    data = data[data["group"] == "controlled_paired"].copy()
    data["shift"] = data["target"].map(readable_shift_name)
    data = data.sort_values("domain_classifier_roc_auc")

    colours = [
        "#0072B2" if name == "Realistic Mixture" else "#7DB7D8"
        for name in data["shift"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.5), sharey=True)

    axes[0].barh(data["shift"], data["mmd_squared"], color=colours)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("MMD squared (log scale)")
    axes[0].set_title("Feature-distribution distance", fontweight="bold")

    axes[1].barh(data["shift"], data["domain_classifier_roc_auc"], color=colours)
    axes[1].axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xlim(0.45, 1.01)
    axes[1].set_xlabel("Domain-classifier ROC AUC")
    axes[1].set_title("Source/target separability", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES / "drift_verification.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    apply_style()
    plot_temporal_performance()
    plot_drift_verification()


if __name__ == "__main__":
    main()
