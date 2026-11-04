"""Classify satellite images with a saved v2 model and optionally save its attention map.

Example::  python -m fyp_drift.evaluation.fmow_v2.predict models/fmow_v2/soft_gate site.jpg --box 0.5,0.5,0.3,0.2 --map map.png
"""

from __future__ import annotations

import argparse
import time

from fyp_drift.evaluation.fmow_v2.models import Predictor


def save_map(image_path, weights, out, box=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fyp_drift.evaluation.fmow_v2.models import IMAGE_SIZE, load_image
    from scipy.ndimage import zoom

    image = load_image(image_path, box)[0].permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(image.mean(-1), cmap="gray")
    ax.imshow(zoom(weights, IMAGE_SIZE / weights.shape[0], order=1), cmap="Oranges", alpha=0.6)
    ax.axis("off")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="folder with model.safetensors and config.json")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--box", help="object box cx,cy,w,h as fractions of the image (crops like training)")
    parser.add_argument("--map", help="save the attention map of the first image to this PNG")
    args = parser.parse_args(argv)
    box = tuple(float(v) for v in args.box.split(",")) if args.box else None
    start = time.time()
    predictor = Predictor.load(args.model)
    print(f"loaded {predictor.config['kind']} in {time.time() - start:.1f}s")
    for path in args.images:
        start = time.time()
        result = predictor.predict(path, box)
        top = ", ".join(f"{name} {p:.2f}" for name, p in result["top"])
        print(f"{path}: {top}  ({1000 * (time.time() - start):.0f} ms)")
        if args.map and "weights" in result:
            try:
                save_map(path, result["weights"], args.map, box)
                print(f"attention map saved to {args.map}")
            except ImportError:
                print("install matplotlib and scipy to save attention maps")
            args.map = None


if __name__ == "__main__":
    main()
