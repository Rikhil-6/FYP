"""Compare matched ViT heads on CIFAR-10 and its recollected test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import Dinov2Model

from fyp_drift.data.clear100 import MEAN, STD
from fyp_drift.evaluation.drift_metrics import distribution_shift_metrics
from fyp_drift.evaluation.wilds_replication import (
    classification_scores,
    extract_features,
    fit_linear_probe,
    stratified_sample,
    train_architecture,
)
from fyp_drift.models.vision_transformer import BACKBONE


CIFAR101_BASE_URL = (
    "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/"
    "master/datasets"
)
CIFAR10_MIRROR_URL = (
    "https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/"
    "resolve/main/cifar-10-python.tar.gz"
)


class WithMetadata(Dataset):
    """Return the metadata placeholder expected by the shared evaluation code."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = np.asarray(dataset.targets, dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, torch.tensor([0])


class CIFAR101(Dataset):
    """Load the class-balanced CIFAR-10.1 v6 release from NumPy files."""

    def __init__(self, root, transform):
        root.mkdir(parents=True, exist_ok=True)
        data_path = root / "cifar10.1_v6_data.npy"
        labels_path = root / "cifar10.1_v6_labels.npy"
        for path in (data_path, labels_path):
            if not path.exists():
                urlretrieve(f"{CIFAR101_BASE_URL}/{path.name}", path)

        self.data = np.load(data_path)
        self.labels = np.load(labels_path).astype(np.int64)
        if self.data.shape != (2000, 32, 32, 3):
            raise ValueError(f"Unexpected CIFAR-10.1 image shape: {self.data.shape}")
        if self.labels.shape != (2000,):
            raise ValueError(f"Unexpected CIFAR-10.1 label shape: {self.labels.shape}")
        if not np.array_equal(np.bincount(self.labels, minlength=10), np.full(10, 200)):
            raise ValueError("CIFAR-10.1 v6 should contain 200 examples per class")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        return self.transform(image), int(self.labels[index]), torch.tensor([1])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Screen matched ViT heads on CIFAR-10 and CIFAR-10.1."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--permutation-limit", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--faithfulness-weight", type=float, default=0.1)
    return parser.parse_args(argv)


def image_transform():
    return Compose(
        [
            Resize((224, 224), antialias=True),
            ToTensor(),
            Normalize(mean=MEAN, std=STD),
        ]
    )


def sampled_loader(dataset, limit, seed, batch_size, workers):
    indices = stratified_sample(dataset.labels, limit, seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.train_limit < 100 or args.eval_limit < 100:
        raise ValueError("The train and evaluation limits must each be at least 100")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = image_transform()
    # The original Toronto host is rate-limited on the project server. This
    # mirror serves the unmodified archive; torchvision still checks its MD5.
    CIFAR10.url = CIFAR10_MIRROR_URL
    datasets = {
        "train": WithMetadata(
            CIFAR10(args.data_root, train=True, download=True, transform=transform)
        ),
        "id": WithMetadata(
            CIFAR10(args.data_root, train=False, download=True, transform=transform)
        ),
        "ood": CIFAR101(args.data_root / "cifar-10.1", transform),
    }
    loaders = {
        "train": sampled_loader(
            datasets["train"], args.train_limit, args.seed, args.batch_size, args.workers
        ),
        "id": sampled_loader(
            datasets["id"],
            args.eval_limit,
            args.seed + 1,
            args.batch_size,
            args.workers,
        ),
        "ood": sampled_loader(
            datasets["ood"],
            args.eval_limit,
            args.seed + 2,
            args.batch_size,
            args.workers,
        ),
    }

    backbone = Dinov2Model.from_pretrained(BACKBONE).to(device).eval()
    extracted = {
        name: extract_features(backbone, data_loader, device, name)
        for name, data_loader in loaders.items()
    }
    del backbone
    if device == "cuda":
        torch.cuda.empty_cache()

    probe = fit_linear_probe(*extracted["train"], args.seed)
    id_scores = classification_scores(probe, *extracted["id"])
    ood_scores = classification_scores(probe, *extracted["ood"])
    drift = distribution_shift_metrics(
        extracted["id"][0],
        extracted["ood"][0],
        permutations=args.permutations,
        permutation_limit=args.permutation_limit,
        seed=args.seed,
    )
    architecture = {
        method: train_architecture(method, 10, loaders, args, device)
        for method in ("baseline", "xai")
    }
    architecture["xai_minus_baseline_ood_macro_f1_points"] = 100 * (
        architecture["xai"]["out_of_distribution"]["macro_f1"]
        - architecture["baseline"]["out_of_distribution"]["macro_f1"]
    )

    payload = {
        "dataset": "CIFAR-10.1 v6",
        "source_dataset": "CIFAR-10",
        "feature_extractor": BACKBONE,
        "seed": args.seed,
        "limits": {
            "training": len(loaders["train"].dataset),
            "in_distribution_test": len(loaders["id"].dataset),
            "recollected_test": len(loaders["ood"].dataset),
        },
        "linear_probe": {
            "in_distribution": id_scores,
            "recollected": ood_scores,
            "macro_f1_change_percentage_points": 100
            * (ood_scores["macro_f1"] - id_scores["macro_f1"]),
        },
        "drift_detection": drift,
        "architecture_screen": architecture,
        "scope": (
            "CIFAR-10.1 is a naturally recollected reproduction test. It tests "
            "dataset shift across collections, not temporal adaptation within a "
            "deployment and not complex industrial imagery."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "cifar10_1_replication.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
