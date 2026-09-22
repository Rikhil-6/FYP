"""FMoW-WILDS ingestion with real acquisition-year metadata and official splits."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as F
from wilds import get_dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def transform(image):
    image = F.resize(image.convert("RGB"), (224, 224))
    return F.normalize(F.to_tensor(image), MEAN, STD)


def open_dataset(root="data", download=False):
    return get_dataset(dataset="fmow", root_dir=str(Path(root)),
                       split_scheme="official", download=download)


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
    """Select actual acquisition years from one official split."""
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
