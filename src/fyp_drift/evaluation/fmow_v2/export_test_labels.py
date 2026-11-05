"""Export the 2016-17 test labels (class id, cloud cover, seen-site flag) so results can be rebuilt without the images.

Example::  python -m fyp_drift.evaluation.fmow_v2.export_test_labels    # writes results/fmow_v2/test_labels.npz
"""

from pathlib import Path

import numpy as np

from fyp_drift.evaluation.fmow_v2.context import load_context
from fyp_drift.evaluation.fmow_v2.site_overlap import seen_site_mask


def main():
    ctx = load_context()
    rows = ctx.splits["future_test"]
    out = Path(__file__).resolve().parents[4] / "results/fmow_v2/test_labels.npz"
    np.savez_compressed(
        out,
        class_id=np.array([r["class_id"] for r in rows], dtype=np.int16),
        cloud_cover=np.array([float(r.get("cloud_cover", 0) or 0) for r in rows], dtype=np.float32),
        seen_site=np.array(seen_site_mask(ctx.splits), dtype=bool),
        key=np.array([r.get("key", "") for r in rows]),
    )
    print("wrote", out, len(rows), "test images")


if __name__ == "__main__":
    main()

