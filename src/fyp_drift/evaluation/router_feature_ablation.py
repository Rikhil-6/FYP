"""Compare router evidence sets on one frozen, corrected-split model pair.

The source run supplies the historical/updated checkpoints and the logged future
features. Only the disjoint 2011 calibration images require model inference.
Neither model is trained or modified by this command.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from fyp_drift.data.clear100 import open_dataset
from fyp_drift.models.routing import FEATURE_SETS, FEATURES, feature_tensor, fit_policy
from fyp_drift.training.temporal_router import (
    build_router,
    build_temporal_loaders,
    seed_everything,
    validate_historical_checkpoint,
)

DEFAULT_COVERAGES = (0.10, 0.25, 0.50)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Refit three routing feature sets on one frozen model pair."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--coverages",
        type=float,
        nargs="+",
        default=DEFAULT_COVERAGES,
        help="Fixed shares of disagreeing predictions routed in the ranking diagnostic.",
    )
    args = parser.parse_args(argv)
    if any(not 0 <= value <= 1 for value in args.coverages):
        parser.error("Every coverage must be between 0 and 1")
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_config(run_dir):
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("calibration_protocol") != "disjoint_class_stratified":
        raise ValueError("Source run must use disjoint class-stratified calibration")
    if config.get("reuse_calibration_for_selection"):
        raise ValueError("Source run reused calibration labels for checkpoint selection")
    if not config.get("long_horizon"):
        raise ValueError("Source run must use the 2010/2011/2012-2014 long horizon")
    return config


@torch.no_grad()
def _calibration_arrays(router, loader, device):
    router.eval()
    features, truth, old, new = [], [], [], []
    for images, labels, _ in tqdm(loader, desc="2011 calibration", unit="batch"):
        comparison = router.compare(images.to(device))
        features.append(feature_tensor(comparison).cpu().numpy())
        truth.append(labels.numpy())
        old.append(comparison["old_logits"].argmax(1).cpu().numpy())
        new.append(comparison["new_logits"].argmax(1).cpu().numpy())
    return tuple(np.concatenate(parts) for parts in (features, truth, old, new))


def _fit_policies(features, truth, old, new):
    beneficial = ((old != truth) & (new == truth)).astype(np.int64)
    return {
        name: fit_policy(
            features, beneficial, old, new, truth, feature_names=feature_names
        )
        for name, feature_names in FEATURE_SETS.items()
    }


def _read_future_rows(path):
    required = {
        "scenario", "year", "dataset_index", "true_class", "old_class",
        "new_class", "routed_to_new", *FEATURES,
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Future decisions are missing columns: {sorted(missing)}")
        groups = defaultdict(list)
        seen = set()
        for row in reader:
            key = (row["scenario"], int(row["year"]), int(row["dataset_index"]))
            if key in seen:
                raise ValueError(f"Duplicate future decision: {key}")
            seen.add(key)
            groups[key[:2]].append(row)
    if not groups:
        raise ValueError("Future decision log is empty")
    return groups


def _arrays(rows):
    features = np.asarray(
        [[float(row[name]) for name in FEATURES] for row in rows], dtype=np.float32
    )
    truth = np.asarray([int(row["true_class"]) for row in rows], dtype=np.int64)
    old = np.asarray([int(row["old_class"]) for row in rows], dtype=np.int64)
    new = np.asarray([int(row["new_class"]) for row in rows], dtype=np.int64)
    recorded = np.asarray(
        [row["routed_to_new"].lower() == "true" for row in rows], dtype=bool
    )
    return features, truth, old, new, recorded


def _policy_scores(features, policy):
    """Reproduce apply_policy's float32 score from a logged feature matrix."""
    selected = [FEATURES.index(name) for name in policy["feature_names"]]
    tensor = torch.from_numpy(features[:, selected])
    mean = tensor.new_tensor(policy["mean"])
    scale = tensor.new_tensor(policy["scale"])
    weights = tensor.new_tensor(policy["coefficients"])
    return torch.sigmoid(
        (((tensor - mean) / scale) * weights).sum(1) + policy["intercept"]
    ).numpy()


def _native_route(features, old, new, policy):
    probabilities = _policy_scores(features, policy)
    eligible = old != new
    if policy["saliency_floor"] is not None:
        eligible &= features[:, FEATURES.index("saliency_js")] >= policy[
            "saliency_floor"
        ]
    return probabilities, eligible & (
        probabilities >= policy["probability_threshold"]
    )


def _policy_reproduction(features, old, new, source, refitted):
    if source["feature_names"] != refitted["feature_names"]:
        raise ValueError("Refitted and source policies use different feature sets")
    _, source_routes = _native_route(features, old, new, source)
    _, refitted_routes = _native_route(features, old, new, refitted)
    source_floor, refitted_floor = source["saliency_floor"], refitted["saliency_floor"]
    floors_match = (source_floor is None and refitted_floor is None) or (
        source_floor is not None
        and refitted_floor is not None
        and abs(source_floor - refitted_floor) <= 1e-4
    )
    parameter_differences = [
        abs(source["intercept"] - refitted["intercept"]),
        *np.abs(np.asarray(source["mean"]) - refitted["mean"]),
        *np.abs(np.asarray(source["scale"]) - refitted["scale"]),
        *np.abs(np.asarray(source["coefficients"]) - refitted["coefficients"]),
    ]
    return {
        "feature_names_match": True,
        "max_parameter_abs_difference": float(max(parameter_differences)),
        "threshold_match": bool(
            source["probability_threshold"] == refitted["probability_threshold"]
        ),
        "floor_match_within_1e-4": bool(floors_match),
        "calibration_route_mismatches": int(np.sum(source_routes != refitted_routes)),
        "calibration_rows": len(features),
    }


def _fixed_coverage_route(probabilities, old, new, coverage):
    """Use a label-free route quota and a stable probability ranking."""
    candidates = np.flatnonzero(old != new)
    count = round(len(candidates) * coverage)
    routed = np.zeros(len(old), dtype=bool)
    if count:
        order = np.lexsort((candidates, -probabilities[candidates]))
        routed[candidates[order[:count]]] = True
    return routed


def _summary(truth, old, new, routed, num_classes):
    prediction = np.where(routed, new, old)

    def score(values):
        support = np.bincount(truth, minlength=num_classes)
        predicted = np.bincount(values, minlength=num_classes)
        true_positive = np.bincount(
            truth[truth == values], minlength=num_classes
        )
        denominator = support + predicted
        f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros(num_classes, dtype=float),
            where=denominator > 0,
        )
        active = denominator > 0
        return {
            "accuracy": float(np.mean(truth == values)),
            "macro_f1": float(f1[active].mean()),
        }

    old_score, new_score, routed_score = score(old), score(new), score(prediction)
    disagreements = int(np.sum(old != new))
    return {
        "samples": len(truth),
        "disagreements": disagreements,
        "routed_to_new": int(np.sum(routed)),
        "disagreement_coverage": float(np.sum(routed) / disagreements)
        if disagreements else 0.0,
        "old": old_score,
        "new": new_score,
        "routed": routed_score,
        "routed_minus_old_f1_pp": 100 * (
            routed_score["macro_f1"] - old_score["macro_f1"]
        ),
        "routed_minus_new_f1_pp": 100 * (
            routed_score["macro_f1"] - new_score["macro_f1"]
        ),
        "beneficial_available": int(np.sum((old != truth) & (new == truth))),
        "beneficial_taken": int(
            np.sum(routed & (old != truth) & (new == truth))
        ),
        "harmful_available": int(np.sum((old == truth) & (new != truth))),
        "harmful_taken": int(np.sum(routed & (old == truth) & (new != truth))),
    }


def _evaluate_groups(groups, policies, source_policy, coverages, num_classes):
    results = {}
    reproduction_mismatches = 0
    total_rows = 0
    for (scenario, year), rows in sorted(groups.items()):
        features, truth, old, new, recorded = _arrays(rows)
        _, reproduced = _native_route(features, old, new, source_policy)
        reproduction_mismatches += int(np.sum(reproduced != recorded))
        total_rows += len(rows)
        condition = {
            "native_policy": {},
            "fixed_disagreement_coverage": {},
        }
        for name, policy in policies.items():
            probabilities, route = _native_route(features, old, new, policy)
            condition["native_policy"][name] = _summary(
                truth, old, new, route, num_classes
            )
            for coverage in coverages:
                key = f"{coverage:.2f}"
                matched = _fixed_coverage_route(
                    probabilities, old, new, coverage
                )
                condition["fixed_disagreement_coverage"].setdefault(key, {})[
                    name
                ] = _summary(truth, old, new, matched, num_classes)
        results.setdefault(scenario, {})[str(year)] = condition
    return results, {"mismatches": reproduction_mismatches, "rows": total_rows}


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Ablation output already exists: {args.output}")
    run_dir = args.run_dir.resolve()
    config = _source_config(run_dir)
    config_path = run_dir / "config.json"
    decisions_path = run_dir / "logs" / "drift_routing_decisions.csv"
    source_policy_path = run_dir / "metrics" / "router_policy.json"
    updated_checkpoint = run_dir / "models" / "best.pt"
    for path in (decisions_path, source_policy_path, updated_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    seed = int(config["seed"])
    seed_everything(seed)
    base_checkpoint = Path(config["base_checkpoint"])
    shared_uniform = bool(config.get("initialize_saliency_from_uniform", False))
    validate_historical_checkpoint(
        base_checkpoint, True, allow_uniform=shared_uniform
    )
    data_root = args.data_root or Path(config.get("data_root", "data"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = open_dataset(data_root)
    loaders = build_temporal_loaders(dataset, argparse.Namespace(**config))
    if loaders.calibration_protocol != "disjoint_class_stratified":
        raise ValueError("Reconstructed calibration split is not disjoint")
    router, _, num_classes, _ = build_router(
        base_checkpoint, device, config["adaptation_profile"],
        initialize_saliency_from_uniform=shared_uniform,
    )
    router.new.load_state_dict(
        torch.load(updated_checkpoint, map_location=device, weights_only=True)
    )
    features, truth, old, new = _calibration_arrays(
        router, loaders.calibration, device
    )
    policies = _fit_policies(features, truth, old, new)
    groups = _read_future_rows(decisions_path)
    source_policy = json.loads(source_policy_path.read_text(encoding="utf-8"))
    source_feature_set = config.get("router_features", "combined")
    if source_feature_set not in policies:
        raise ValueError(f"Unknown source router feature set: {source_feature_set}")
    policy_reproduction = _policy_reproduction(
        features, old, new, source_policy, policies[source_feature_set]
    )
    if (
        policy_reproduction["calibration_route_mismatches"]
        or not policy_reproduction["threshold_match"]
        or not policy_reproduction["floor_match_within_1e-4"]
        or policy_reproduction["max_parameter_abs_difference"] > 1e-4
    ):
        raise ValueError(
            "Refitted policy does not reproduce the source policy on 2011 "
            f"calibration rows: {policy_reproduction}"
        )
    conditions, reproduction = _evaluate_groups(
        groups, policies, source_policy, args.coverages, num_classes
    )
    if reproduction["mismatches"]:
        raise ValueError(
            "Logged future features fail to reproduce the source policy decisions "
            f"({reproduction['mismatches']}/{reproduction['rows']}); check provenance"
        )

    output = {
        "source_run": str(run_dir),
        "training_performed": False,
        "calibration_protocol": loaders.calibration_protocol,
        "calibration_samples": len(truth),
        "calibration_beneficial_switches": int(
            np.sum((old != truth) & (new == truth))
        ),
        "source_policy_refit_reproduction": policy_reproduction,
        "model_pair": {
            "historical_checkpoint": str(base_checkpoint),
            "updated_checkpoint": str(updated_checkpoint),
        },
        "future_features": {
            "path": str(decisions_path),
            "sha256": _sha256(decisions_path),
            "rounded_decimals": 6,
            "source_policy_reproduction": reproduction,
            "replay_exact": reproduction["mismatches"] == 0,
        },
        "calibration_config_sha256": _sha256(config_path),
        "policies": policies,
        "fixed_coverage_note": (
            "Transductive diagnostic ranking at a prespecified share of old/new "
            "disagreements; all feature sets route the same count, without "
            "saliency floors or future-label threshold fitting. It uses the full "
            "unlabelled test group to set the quota, so it is not a deployable "
            "streaming policy. It is separate from the native policy."
        ),
        "conditions": conditions,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "router_feature_ablation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "calibration_samples": len(truth),
                "source_policy_refit_reproduction": policy_reproduction,
                "source_policy_reproduction": reproduction,
                "conditions": {
                    scenario: list(years) for scenario, years in conditions.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
