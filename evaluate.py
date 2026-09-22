"""Classification metrics by temporal split, upload year, and class."""
import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from data import make_loaders
from metrics import grouped_classification
from model import TemporalViT


@torch.no_grad()
def evaluate(model, loaders, year_column, device, splits=("id_val", "id_test", "val", "test")):
    model.eval()
    report, records = {}, []
    for split in splits:
        rows = []
        for images, labels, metadata in tqdm(loaders[split], desc=f"evaluate {split}", unit="batch", leave=False):
            probabilities = model(images.to(device))[0].softmax(1)
            confidence, prediction = probabilities.max(1)
            for year, actual, predicted, score in zip(metadata[:, year_column].tolist(),
                                                       labels.tolist(), prediction.cpu().tolist(),
                                                       confidence.cpu().tolist()):
                rows.append({"split": split, "year": 2002 + year if year >= 0 else -1,
                             "true_class": actual, "prediction": predicted,
                             "confidence": round(score, 6)})
        report[split] = grouped_classification(rows)
        records.extend(rows)
    return report, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("baseline", "xai"), required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TemporalViT(args.method, num_classes=checkpoint["classifier.weight"].shape[0]).to(device)
    model.load_state_dict(checkpoint)
    loaders, year_column = make_loaders(args.data_root, args.batch_size, limit_eval=args.limit_eval)
    report, records = evaluate(model, loaders, year_column, device)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2))
        with (args.output / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps({split: {"accuracy": values["all"]["accuracy"],
                              "macro_f1": values["all"]["macro_f1"],
                              "macro_iou": values["all"]["macro_iou"]}
                      for split, values in report.items()}, indent=2))


if __name__ == "__main__":
    main()
