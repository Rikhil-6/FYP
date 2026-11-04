"""Train the seed-42 historical head copy and the attention, hard-gate and soft-gate heads, and save them for inference.

Heads train on the cached clean tokens exactly as in the experiments (lr 1e-3, 15 epochs, best validation epoch).
Example::  python -m fyp_drift.evaluation.fmow_v2.export_heads --out models/fmow_v2
"""

from __future__ import annotations

import argparse
import os
import types
from pathlib import Path

import numpy as np
import torch
from fyp_drift.evaluation import saliency_attention_study as sas
from fyp_drift.evaluation import soft_saliency_heads as soft
from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_MANIFEST,
    load_rows,
    log,
    scores,
    study_splits,
)
from fyp_drift.evaluation.fmow_v2.context import WIDTH
from fyp_drift.evaluation.fmow_v2.models import save_model

HEADS = {"attention": "attention", "hard_gate": "gated", "soft_gate": "soft_gated"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("models/fmow_v2"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    runs, tokens = Path(os.environ["FYP_RUNS"]), Path(os.environ["FYP_TOKENS"])
    rows, classes = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    names = sorted({r["category"] for r in rows})
    data = {
        s: (np.load(tokens / f"clean_{s}.npy", mmap_mode="r"), np.array([r["class_id"] for r in splits[s]]))
        for s in sas.SPLITS
    }
    historical = torch.load(runs / "main/head.pt", weights_only=True)
    save_model(args.out / "historical", "historical", historical, names)
    settings = types.SimpleNamespace(lr=1e-3, epochs=15, batch=128)
    for name, kind in HEADS.items():
        if kind == "soft_gated":
            model, best = soft.train(kind, args.seed, historical, WIDTH, classes, data, settings)
        else:
            model, best = sas.train(kind, args.seed, historical, WIDTH, classes, data, 0.0, settings)
        predictions = sas.infer(model, *data["future_test"])[0]
        test_f1 = scores(data["future_test"][1], predictions, classes)["macro_f1"]
        save_model(
            args.out / name, kind, model.state_dict(), names, seed=args.seed, val_f1=best["val_f1"], test_f1=test_f1
        )
        log(f"exported {name}: val={best['val_f1']:.4f} test={test_f1:.4f}")


if __name__ == "__main__":
    main()
