import argparse
import csv
import json
from pathlib import Path


def descending_ranks(values):
    values = list(values)
    ordered = sorted(set(values), reverse=True)
    return {value: ordered.index(value) + 1 for value in values}


def rank_sources(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["group"] == "controlled_paired"
            and not row["target"].startswith("realistic_mix")
        ]

    mmd_ranks = descending_ranks(float(row["mmd_squared"]) for row in rows)
    auc_ranks = descending_ranks(
        float(row["domain_classifier_roc_auc"]) for row in rows
    )
    ranking = []
    for row in rows:
        mmd = float(row["mmd_squared"])
        auc = float(row["domain_classifier_roc_auc"])
        mmd_rank = mmd_ranks[mmd]
        auc_rank = auc_ranks[auc]
        ranking.append(
            {
                "source": row["target"].rsplit("_severity_", 1)[0],
                "mmd_squared": mmd,
                "domain_classifier_roc_auc": auc,
                "mmd_permutation_p": float(row["mmd_permutation_p"]),
                "mmd_rank": mmd_rank,
                "auc_rank": auc_rank,
                "mean_rank": (mmd_rank + auc_rank) / 2,
            }
        )
    return sorted(ranking, key=lambda row: (row["mean_rank"], row["mmd_rank"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ranking = rank_sources(args.csv_path)
    payload = {
        "criterion": "mean descending rank across MMD squared and domain AUC",
        "top_n": args.top_n,
        "selected_sources": [row["source"] for row in ranking[: args.top_n]],
        "ranking": ranking,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
