"""Shared run context (splits, image cache, cloud settings, frozen head), configured by scripts/fmow_v2/env.sh."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from fyp_drift.evaluation import fmow_dinov2l_study as study
from torch import nn

BACKBONE = "facebook/dinov2-large"
WIDTH = 1024  # DINOv2-L token width


@dataclass
class Context:
    runs: Path  # root of the v2 outputs (main/, saliency_attention/, ...)
    rows: list  # manifest rows with class ids, crops and acquisition metadata
    classes: int
    splits: dict  # hist_train, hist_val, adapt_train, adapt_val, future_test
    cache: object  # memory-mapped uint8 image cache
    overlay_alpha: float  # training-overlay opacity chosen on 2010-13 validation
    cumulus: dict  # {"off_object": (strength, "off_object"), "on_object": (strength, "on_object")}

    def historical_head(self) -> nn.Linear:
        """The frozen 2010-13 linear head (on the GPU, gradients off)."""
        state = torch.load(self.runs / "main/head.pt", weights_only=True)
        head = nn.Linear(state["weight"].shape[1], self.classes).cuda().eval()
        head.load_state_dict(state)
        head.requires_grad_(False)
        return head

    def test_conditions(self) -> dict:
        """Every test condition as (cloud strength, cloud region): clean, training overlay and held-out cumulus."""
        return {"clean": (0.0, None), "overlay": (self.overlay_alpha, None), **self.cumulus}


def load_context(limit: int = 0) -> Context:
    """Load the manifest, splits and cache; ``limit`` caps every split for a quick smoke test."""
    runs = Path(os.environ["FYP_RUNS"])
    rows, classes = study.load_rows(study.DEFAULT_MANIFEST)
    splits = study.study_splits(rows)
    if limit:
        splits = {name: part[:limit] for name, part in splits.items()}
    cache = study.image_cache(rows, study.DEFAULT_CACHE, 4)
    overlay_alpha = json.loads((runs / "main/calib.json").read_text())["selected_alpha"]
    regions = json.loads((runs / "main/calib_regions.json").read_text())
    cumulus = {
        name: (cfg["selected_alpha"], name)
        for name, cfg in regions.items()
        if isinstance(cfg, dict) and "selected_alpha" in cfg
    }
    return Context(runs, rows, classes, splits, cache, overlay_alpha, cumulus)


def pool_2x2(tokens: torch.Tensor, grid: int = 20) -> torch.Tensor:
    """Average neighbouring 2 x 2 patch tokens: (B, grid*grid, D) -> (B, (grid/2)^2, D), row-major."""
    batch, _, width = tokens.shape
    half = grid // 2
    return tokens.reshape(batch, half, 2, half, 2, width).mean((2, 4)).reshape(batch, half * half, width)


def trainable(module: nn.Module) -> list:
    return [p for p in module.parameters() if p.requires_grad]
