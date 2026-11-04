"""Cache frozen 10x10 cell tokens for the held-out cumulus test conditions, so heads can be tested on them.

Example::  python -m fyp_drift.evaluation.fmow_v2.heldout_tokens
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from fyp_drift.evaluation import fmow_dinov2l_study as study
from fyp_drift.evaluation.fmow_v2.context import BACKBONE, WIDTH, load_context, pool_2x2


def main():
    ctx = load_context()
    out_dir = Path(os.environ.get("FYP_TOKENS_EXTRA", "/mnt/sdd/FYP/cache/adapter_tokens_v2p_extra"))
    out_dir.mkdir(exist_ok=True)
    backbone = study.Backbone(BACKBONE).cuda().eval()
    rows = ctx.splits["future_test"]
    for name, (strength, region) in ctx.cumulus.items():
        path = out_dir / f"{name}_future_test.npy"
        if path.with_suffix(".done").exists():
            continue
        cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(rows), 100, WIDTH))
        written = 0
        with torch.no_grad():
            for images, _, _ in study.image_loader(ctx.cache, rows, 64, 4, strength, cloud_region=region):
                with torch.autocast("cuda", dtype=torch.float16):
                    tokens = backbone(images.cuda(non_blocking=True))[:, 1:].float()
                cells = pool_2x2(tokens)
                cache[written : written + len(cells)] = cells.half().cpu().numpy()
                written += len(cells)
        assert written == len(rows)
        cache.flush()
        path.with_suffix(".done").write_text("ok")
        study.log("cached held-out tokens", name, strength)


if __name__ == "__main__":
    main()
