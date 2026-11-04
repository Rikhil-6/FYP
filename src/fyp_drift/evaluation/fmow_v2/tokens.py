"""Build the frozen-token caches: 2x2 pooling, the cloud-overlay condition and the held-out cumulus conditions.

Example::  python -m fyp_drift.evaluation.fmow_v2.tokens cloud    # then: pool, heldout
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from fyp_drift.evaluation import fmow_dinov2l_study as study
from fyp_drift.evaluation.fmow_v2.context import BACKBONE, WIDTH, load_context, pool_2x2
from fyp_drift.evaluation.saliency_adapters import SPLITS, token_cache


def pool():
    """Average 2x2 patch tokens (20x20 -> 10x10) so the head-training caches fit in RAM."""
    src, dst = (
        Path("/mnt/sdd/FYP/cache/adapter_tokens_v2"),
        Path("/mnt/sdd/FYP/cache/adapter_tokens_v2p"),
    )
    dst.mkdir(exist_ok=True)
    for f in sorted(src.glob("*.npy")):
        out = dst / f.name
        if out.with_suffix(".done").exists():
            continue
        x = np.load(f, mmap_mode="r")
        n, p, d = x.shape
        g = int(round(p**0.5))
        assert g * g == p and g % 2 == 0
        y = np.lib.format.open_memmap(
            out, mode="w+", dtype=np.float16, shape=(n, (g // 2) ** 2, d)
        )
        for s in range(0, n, 256):
            b = (
                np.asarray(x[s : s + 256], dtype=np.float32)
                .reshape(-1, g // 2, 2, g // 2, 2, d)
                .mean((2, 4))
            )
            y[s : s + 256] = b.reshape(len(b), -1, d).astype(np.float16)
        y.flush()
        out.with_suffix(".done").write_text("ok")
        print("pooled", f.name, x.shape, "->", y.shape, flush=True)


def cloud():
    """Cache 20x20 tokens for every split under the training cloud overlay (clean caches come from saliency_adapters)."""
    R, T = (Path(os.environ["FYP_RUNS"]), Path(os.environ["FYP_TOKENS"]))
    rows, _ = study.load_rows(study.DEFAULT_MANIFEST)
    splits = study.study_splits(rows)
    cache = study.image_cache(rows, study.DEFAULT_CACHE, 4)
    alpha = json.loads((R / "main/calib.json").read_text())["selected_alpha"]
    backbone = study.Backbone(BACKBONE).cuda().eval()
    for split in SPLITS:
        token_cache(backbone, cache, splits[split], T / f"cloud_{split}.npy", alpha, 4)
        study.log("tokens ready cloud", split)


def heldout():
    """Cache 10x10 cell tokens of the test images under each held-out cumulus condition."""
    ctx = load_context()
    out_dir = Path(
        os.environ.get(
            "FYP_TOKENS_EXTRA", "/mnt/sdd/FYP/cache/adapter_tokens_v2p_extra"
        )
    )
    out_dir.mkdir(exist_ok=True)
    backbone = study.Backbone(BACKBONE).cuda().eval()
    rows = ctx.splits["future_test"]
    for name, (strength, region) in ctx.cumulus.items():
        path = out_dir / f"{name}_future_test.npy"
        if path.with_suffix(".done").exists():
            continue
        cache = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(len(rows), 100, WIDTH)
        )
        written = 0
        with torch.no_grad():
            for images, _, _ in study.image_loader(
                ctx.cache, rows, 64, 4, strength, cloud_region=region
            ):
                with torch.autocast("cuda", dtype=torch.float16):
                    tokens = backbone(images.cuda(non_blocking=True))[:, 1:].float()
                cells = pool_2x2(tokens)
                cache[written : written + len(cells)] = cells.half().cpu().numpy()
                written += len(cells)
        assert written == len(rows)
        cache.flush()
        path.with_suffix(".done").write_text("ok")
        study.log("cached held-out tokens", name, strength)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=("cloud", "pool", "heldout"))
    {"cloud": cloud, "pool": pool, "heldout": heldout}[parser.parse_args(argv).step]()


if __name__ == "__main__":
    main()
