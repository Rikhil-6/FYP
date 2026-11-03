"""Cache frozen tokens for the cloud-overlay condition (clean caches come from saliency_adapters)."""

import json
import os
from pathlib import Path

from fyp_drift.evaluation.fmow_dinov2l_study import (
    DEFAULT_CACHE,
    DEFAULT_MANIFEST,
    Backbone,
    image_cache,
    load_rows,
    log,
    study_splits,
)
from fyp_drift.evaluation.saliency_adapters import SPLITS, token_cache


def main():
    R, T = (Path(os.environ["FYP_RUNS"]), Path(os.environ["FYP_TOKENS"]))
    rows, _ = load_rows(DEFAULT_MANIFEST)
    splits = study_splits(rows)
    cache = image_cache(rows, DEFAULT_CACHE, 4)
    alpha = json.loads((R / "main/calib.json").read_text())["selected_alpha"]
    backbone = Backbone("facebook/dinov2-large").cuda().eval()
    for split in SPLITS:
        token_cache(backbone, cache, splits[split], T / f"cloud_{split}.npy", alpha, 4)
        log("tokens ready cloud", split)


if __name__ == "__main__":
    main()
