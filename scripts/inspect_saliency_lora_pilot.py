"""Summarize parameter changes between saved matched LoRA pilot arms."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    states = {
        mode: torch.load(args.run_dir / f"{mode}.pt", map_location="cpu", weights_only=True)
        for mode in ("none", "uniform", "shuffled", "saliency")
    }
    keys = [key for key in states["none"] if ".up.weight" in key or
            ".down.weight" in key or key.startswith("classifier.")]
    output = {}
    for mode in states:
        state = states[mode]
        output[mode] = {
            "lora_up_l2": sum(float(state[key].square().sum()) for key in keys
                              if ".up.weight" in key) ** 0.5,
            "difference_from_none_l2": sum(float(
                (state[key] - states["none"][key]).square().sum()
            ) for key in keys) ** 0.5,
            "classifier_difference_from_none_l2": sum(float(
                (state[key] - states["none"][key]).square().sum()
            ) for key in keys if key.startswith("classifier.")) ** 0.5,
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
