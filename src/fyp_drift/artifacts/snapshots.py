"""Manually preserve a selected run checkpoint with its selection metadata."""

import argparse
from pathlib import Path

from fyp_drift.artifacts.run_store import manual_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--which", choices=("best", "last"), default="best")
    args = parser.parse_args()
    print(manual_snapshot(args.run_dir, args.name, args.which))


if __name__ == "__main__":
    main()
