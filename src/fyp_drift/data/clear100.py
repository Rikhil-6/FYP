"""CLEAR-100 images with chronological splits defined in the ingestion manifest."""

import csv
import random
from pathlib import Path
from typing import ClassVar

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 100


def transform(image, index=None):
    image = F.resize(image.convert("RGB"), (224, 224))
    return F.normalize(F.to_tensor(image), MEAN, STD)


class ClearSubset(Dataset):
    def __init__(self, root, rows, image_transform=None):
        self.root = Path(root)
        self.rows = rows
        self.transform = image_transform
        self.metadata_array = torch.tensor(
            [[int(row["year"]) - 2002] for row in rows], dtype=torch.long
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image, index)
        return (
            image,
            torch.tensor(int(row["label"]), dtype=torch.long),
            self.metadata_array[index],
        )


class ClearDataset:
    metadata_fields: ClassVar = ["year"]
    n_classes = NUM_CLASSES

    def __init__(self, root="data"):
        self.root = Path(root).resolve() / "clear100"
        manifest = self.root / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"CLEAR-100 manifest missing: {manifest}. Run 'bash deploy/server.sh ingest'."
            )
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = {
            split: [row for row in rows if row["split"] == split]
            for split in ("train", "id_val", "id_test", "val", "test")
        }
        self.rows["train_source"] = [row for row in rows if row["source"] == "train"]
        self.rows["test_source"] = [row for row in rows if row["source"] == "test"]
        if any(not split_rows for split_rows in self.rows.values()):
            raise ValueError("CLEAR-100 manifest has an empty required split")

    def get_subset(self, split, transform=None):
        return ClearSubset(self.root, self.rows[split], transform)


def open_dataset(root="data", download=False):
    if download:
        raise ValueError("Run 'uv run fyp-ingest' to download CLEAR-100")
    return ClearDataset(root)


def make_loaders(
    root="data", batch_size=8, workers=0, seed=42, limit_train=None, limit_eval=None
):
    dataset = open_dataset(root)
    subsets = {
        name: dataset.get_subset(name, transform=transform)
        for name in ("train", "id_val", "id_test", "val", "test")
    }
    for name, subset in list(subsets.items()):
        limit = limit_train if name == "train" else limit_eval
        if limit:
            if name == "train":
                indices = list(range(min(limit, len(subset))))
            else:
                # Preserve every year represented by a temporal split in small
                # debug runs instead of truncating into its earliest year only.
                by_year = {}
                for index, year in enumerate(subset.metadata_array[:, 0].tolist()):
                    by_year.setdefault(year, []).append(index)
                indices = []
                while len(indices) < min(limit, len(subset)):
                    added = False
                    for year in sorted(by_year):
                        if by_year[year]:
                            indices.append(by_year[year].pop(0))
                            added = True
                            if len(indices) == min(limit, len(subset)):
                                break
                    if not added:
                        break
            subsets[name] = Subset(subset, indices)
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=workers,
            generator=generator if name == "train" else None,
        )
        for name, subset in subsets.items()
    }
    return loaders, dataset.metadata_fields.index("year")


def year_loader(
    dataset,
    split,
    years,
    batch_size=8,
    shuffle=False,
    limit=None,
    seed=42,
    image_transform=transform,
):
    """Select real upload-year buckets from a CLEAR-100 split."""
    subset = dataset.get_subset(split, transform=image_transform)
    year_column = dataset.metadata_fields.index("year")
    year_ids = torch.tensor([year - 2002 for year in years])
    indices = torch.where(torch.isin(subset.metadata_array[:, year_column], year_ids))[
        0
    ].tolist()
    if limit:
        indices = indices[:limit]
    selected = Subset(subset, indices)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    ), year_column


def stratified_year_partition_loaders(
    dataset,
    split,
    year,
    batch_size=8,
    fraction=0.5,
    limit=None,
    seed=42,
    image_transform=transform,
):
    """Create disjoint, class-stratified selection and calibration loaders.

    The first loader is used for checkpoint selection and the second for fitting
    the routing policy. Keeping their examples disjoint prevents the router from
    being calibrated on the same labels that selected the adapted checkpoint.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    subset = dataset.get_subset(split, transform=image_transform)
    year_column = dataset.metadata_fields.index("year")
    year_id = year - 2002
    candidates = torch.where(subset.metadata_array[:, year_column] == year_id)[
        0
    ].tolist()
    by_class = {}
    for index in candidates:
        label = int(subset.rows[index]["label"])
        by_class.setdefault(label, []).append(index)

    selection_indices, calibration_indices = [], []
    for label, indices in sorted(by_class.items()):
        shuffled = list(indices)
        random.Random(seed + 1009 * label).shuffle(shuffled)
        if len(shuffled) == 1:
            # Alternate singleton classes deterministically. CLEAR-100 normally
            # has multiple samples per class, but this keeps debug subsets valid.
            (selection_indices if label % 2 == 0 else calibration_indices).extend(
                shuffled
            )
            continue
        cut = max(1, min(len(shuffled) - 1, round(len(shuffled) * fraction)))
        selection_indices.extend(shuffled[:cut])
        calibration_indices.extend(shuffled[cut:])

    selection_indices.sort()
    calibration_indices.sort()
    if limit:
        selection_indices = selection_indices[:limit]
        calibration_indices = calibration_indices[:limit]
    if not selection_indices or not calibration_indices:
        raise ValueError(
            f"Year {year} cannot be split into non-empty selection and calibration sets"
        )

    return (
        DataLoader(Subset(subset, selection_indices), batch_size=batch_size),
        DataLoader(Subset(subset, calibration_indices), batch_size=batch_size),
        year_column,
    )
