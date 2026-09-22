"""Explicit opt-in download: FMoW-WILDS is a large benchmark."""
import argparse

from data import open_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    dataset = open_dataset(args.data_root, download=True)
    print({split: len(dataset.get_subset(split)) for split in ("train", "id_test", "val", "test")})
    print("metadata fields:", dataset.metadata_fields)


if __name__ == "__main__":
    main()
