"""Test whether a trained saliency map changes predictions on fixed image tokens.

This intervention does not retrain the model or tune on future labels. It
changes patch pooling inside the saved classifier while holding its backbone,
class token, and classifier weights fixed. The class-only and patch-only arms
are diagnostic masks, not separately trained model comparisons.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from fyp_drift.data.clear100 import open_dataset, year_loader
from fyp_drift.data.corruptions import SCENARIOS, drift_transform
from fyp_drift.evaluation.metrics import classification_metrics
from fyp_drift.models.vision_transformer import TemporalViT
from fyp_drift.training.temporal_router import seed_everything

DEFAULT_SCENARIOS = (
    "clean",
    "motion_blur",
    "viewpoint_shift",
    "occlusion",
    "realistic_mix",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=DEFAULT_SCENARIOS)
    parser.add_argument("--role", choices=("historical", "updated"),
                        default="updated")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists():
        parser.error("--output already exists; use a new path")
    return args


def _checkpoint(run_dir, config, role):
    if role == "historical":
        return Path(config["base_checkpoint"])
    return run_dir / "models" / "best.pt"


@torch.no_grad()
def evaluate_condition(model, loader, device, classes):
    model.eval()
    truth = []
    predictions = {}
    true_logit_delta = {}
    for images, labels, _ in tqdm(loader, desc="patch mechanism", unit="batch"):
        logits = model.patch_weight_interventions(images.to(device))
        actual = labels.to(device)
        truth.extend(labels.tolist())
        reference_logit = logits["learned"].gather(1, actual[:, None]).squeeze(1)
        for name, values in logits.items():
            predictions.setdefault(name, []).extend(values.argmax(1).cpu().tolist())
            true_logit_delta.setdefault(name, []).extend(
                (values.gather(1, actual[:, None]).squeeze(1) - reference_logit)
                .cpu().tolist()
            )
    actual = np.asarray(truth)
    reference = np.asarray(predictions["learned"])
    results = {}
    for name, values in predictions.items():
        prediction = np.asarray(values)
        metrics = classification_metrics(actual, prediction, classes=classes)
        results[name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "top1_flip_rate_vs_learned": float(np.mean(prediction != reference)),
            "learned_correct_intervention_wrong": int(
                np.sum((reference == actual) & (prediction != actual))
            ),
            "learned_wrong_intervention_correct": int(
                np.sum((reference != actual) & (prediction == actual))
            ),
            "mean_true_class_logit_delta_vs_learned": float(
                np.mean(true_logit_delta[name])
            ),
        }
    return {"samples": len(truth), "interventions": results}


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("calibration_protocol") != "disjoint_class_stratified":
        raise ValueError("Use a corrected-split source run")
    if not config.get("long_horizon"):
        raise ValueError("Use the 2010/2011/2012-2014 horizon")
    seed = int(config["seed"])
    seed_everything(seed)
    checkpoint = _checkpoint(run_dir, config, args.role)
    pooling_mode = config.get("pooling_mode", "class_plus_patch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = state["classifier.weight"].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalViT("xai", classes, pooling_mode=pooling_mode).to(device)
    model.load_state_dict(state)
    if args.role == "historical" and config.get(
        "initialize_saliency_from_uniform", False
    ):
        with torch.no_grad():
            model.saliency_head[1].weight.zero_()
            model.saliency_head[1].bias.zero_()
    dataset = open_dataset(args.data_root)
    conditions = {}
    for scenario in args.scenarios:
        severity = max(1, args.year - 2011)
        loader, _ = year_loader(
            dataset, "test_source", (args.year,), args.batch_size,
            limit=args.limit, seed=seed,
            image_transform=drift_transform(scenario, severity, seed),
        )
        conditions[scenario] = evaluate_condition(model, loader, device, classes)
        print(json.dumps({"scenario": scenario, "year": args.year,
                          "result": conditions[scenario]}), flush=True)
    report = {
        "source_run": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_role": args.role,
        "pooling_mode": pooling_mode,
        "training_performed": False,
        "future_labels_used_for_tuning": False,
        "year": args.year,
        "seed": seed,
        "sampling_limit": args.limit,
        "interpretation": (
            "Learned, uniform, permuted, and inverted patch weights share the same tokens, "
            "class token, and classifier. Class-only and patch-only mask one "
            "pathway and are not trained baselines. This tests prediction "
            "dependence on patch weights, not object localization or the "
            "generalization benefit of saliency-guided training."
        ),
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
