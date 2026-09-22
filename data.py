"""CLEAR-100 images with chronological splits defined in the ingestion manifest."""
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 100


def transform(image):
    image = F.resize(image.convert("RGB"), (224, 224))
    return F.normalize(F.to_tensor(image), MEAN, STD)


class ClearSubset(Dataset):
    def __init__(self, root, rows, image_transform=None):
        self.root = Path(root)
        self.rows = rows
        self.transform = image_transform
        self.metadata_array = torch.tensor([[int(row["year"]) - 2002] for row in rows], dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, torch.tensor(int(row["label"]), dtype=torch.long), self.metadata_array[index]


class ClearDataset:
    metadata_fields = ["year"]
    n_classes = NUM_CLASSES

    def __init__(self, root="data"):
        self.root = Path(root).resolve() / "clear100"
        manifest = self.root / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"CLEAR-100 manifest missing: {manifest}. Run 'bash deploy/server.sh ingest'.")
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = {split: [row for row in rows if row["split"] == split]
                     for split in ("train", "id_val", "id_test", "val", "test")}
        if any(not split_rows for split_rows in self.rows.values()):
            raise ValueError("CLEAR-100 manifest has an empty required split")

    def get_subset(self, split, transform=None):
        return ClearSubset(self.root, self.rows[split], transform)


def open_dataset(root="data", download=False):
    if download:
        raise ValueError("Use ingest.py for CLEAR-100 downloads")
    return ClearDataset(root)


def make_loaders(root="data", batch_size=8, workers=0, seed=42,
                 limit_train=None, limit_eval=None):
    dataset = open_dataset(root)
    subsets = {name: dataset.get_subset(name, transform=transform)
               for name in ("train", "id_val", "id_test", "val", "test")}
    for name, subset in list(subsets.items()):
        limit = limit_train if name == "train" else limit_eval
        if limit:
            subsets[name] = Subset(subset, range(min(limit, len(subset))))
    generator = torch.Generator().manual_seed(seed)
    loaders = {name: DataLoader(subset, batch_size=batch_size, shuffle=(name == "train"),
                                num_workers=workers, generator=generator if name == "train" else None)
               for name, subset in subsets.items()}
    return loaders, dataset.metadata_fields.index("year")


def year_loader(dataset, split, years, batch_size=8, shuffle=False, limit=None, seed=42):
    """Select real upload-year buckets from a CLEAR-100 split."""
    subset = dataset.get_subset(split, transform=transform)
    year_column = dataset.metadata_fields.index("year")
    year_ids = torch.tensor([year - 2002 for year in years])
    indices = torch.where(torch.isin(subset.metadata_array[:, year_column], year_ids))[0].tolist()
    if limit:
        indices = indices[:limit]
    selected = Subset(subset, indices)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(selected, batch_size=batch_size, shuffle=shuffle,
                      generator=generator if shuffle else None), year_column
