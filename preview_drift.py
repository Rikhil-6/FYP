"""Save a source-labelled grid of the v2 drift transforms before model runs."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data import MEAN, STD, open_dataset
from drift import REALISTIC_BENCHMARK_SCENARIOS, drift_transform, scenario_components


def rgb(tensor):
    value = tensor.cpu() * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    return Image.fromarray((value.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument("--offset", type=int, default=777)
    parser.add_argument("--severity", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = open_dataset(args.data_root)
    reference = dataset.get_subset("test_source", transform=drift_transform("clean", 1, args.seed))
    year_indices = torch.where(reference.metadata_array[:, 0] == args.year - 2002)[0].tolist()
    index = year_indices[args.offset % len(year_indices)]
    panels, records = [], []
    for scenario in REALISTIC_BENCHMARK_SCENARIOS:
        subset = dataset.get_subset(
            "test_source", transform=drift_transform(scenario, args.severity, args.seed))
        image, label, _ = subset[index]
        components = scenario_components(scenario, index, args.severity, args.seed)
        title = scenario.replace("_", " ")
        if scenario == "realistic_mix":
            title += "\n" + ", ".join(name.replace("_", " ") for name in components)
        panels.append((title, rgb(image)))
        records.append({"scenario": scenario, "components": list(components)})

    columns, cell_width, cell_height = 4, 224, 270
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for position, (title, panel) in enumerate(panels):
        left, top = position % columns * cell_width, position // columns * cell_height
        canvas.paste(panel, (left, top + 42))
        draw.multiline_text((left + 4, top + 4), title, fill="black", spacing=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    metadata = {"dataset_index": index, "year": args.year, "severity": args.severity,
                "true_class": int(label), "panels": records}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
