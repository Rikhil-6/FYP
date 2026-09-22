"""Explicit opt-in download: FMoW-WILDS is a large benchmark."""
import argparse
import socket
from pathlib import Path
from urllib.parse import urlsplit

from data import open_dataset
from wilds.datasets.fmow_dataset import FMoWDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    data_dir = Path(args.data_root) / "fmow_v1.1"
    release = data_dir / "RELEASE_v1.1.txt"
    if release.is_file():
        required = (data_dir / "rgb_metadata.csv", data_dir / "country_code_mapping.csv", data_dir / "images")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            parser.exit(2, f"FMoW release marker exists but files are missing: {missing}. "
                           "Repair the dataset before training.\n")
        print(f"FMoW v1.1 already present at {data_dir}; skipping download.", flush=True)
        dataset = open_dataset(args.data_root, download=False)
    else:
        source = FMoWDataset._versions_dict["1.1"]["download_url"]
        hostname = urlsplit(source).hostname
        print(f"Checking FMoW source connectivity: {hostname}:443", flush=True)
        try:
            with socket.create_connection((hostname, 443), timeout=12):
                pass
        except OSError as error:
            parser.exit(2, f"FMoW source {hostname}:443 is unreachable: {error}. "
                           "The dataset download has not started; retry ingestion when the source is reachable.\n")
        dataset = open_dataset(args.data_root, download=True)
    print({split: len(dataset.get_subset(split)) for split in ("train", "id_test", "val", "test")})
    print("metadata fields:", dataset.metadata_fields)


if __name__ == "__main__":
    main()
