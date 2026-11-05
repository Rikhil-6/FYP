"""Cross-period site audit: how many 2016-17 test images come from a site that also occurs in an earlier split?

The splits are grouped by country and UTM zone *within* each period, which does not stop the same physical site (an fMoW
location id such as ``airport_terminal_1093``) from appearing in two periods. This script measures that overlap.

Example::  python -m fyp_drift.evaluation.fmow_v2.site_overlap     # writes results/fmow_v2/site_overlap.json
"""

import json
from pathlib import Path

EARLIER = ("hist_train", "hist_val", "adapt_train", "adapt_val")


def site_of(path):
    """fMoW location id of an image: the name of the folder that holds it (category_locationid)."""
    return Path(path).parent.name


def audit(splits):
    """`splits` maps a split name to its rows (dicts with ``local_path`` and ``group``); returns the overlap counts."""
    sites = {k: {site_of(r["local_path"]) for r in v} for k, v in splits.items()}
    groups = {k: {r["group"] for r in v} for k, v in splits.items()}
    earlier_sites = set().union(*(sites[k] for k in EARLIER))
    earlier_groups = set().union(*(groups[k] for k in EARLIER))
    test = splits["future_test"]
    return dict(
        test_images=len(test),
        test_sites=len(sites["future_test"]),
        test_sites_shared_with={k: len(sites["future_test"] & sites[k]) for k in EARLIER},
        test_sites_shared_with_any_earlier_split=len(sites["future_test"] & earlier_sites),
        test_images_from_shared_sites=sum(site_of(r["local_path"]) in earlier_sites for r in test),
        test_country_utm_groups=len(groups["future_test"]),
        test_groups_also_earlier=len(groups["future_test"] & earlier_groups),
    )


def seen_site_mask(splits):
    """Boolean list over the test rows: True where the image's site occurs in an earlier split."""
    earlier = set().union(*({site_of(r["local_path"]) for r in splits[k]} for k in EARLIER))
    return [site_of(r["local_path"]) in earlier for r in splits["future_test"]]


def main():
    from fyp_drift.evaluation.fmow_dinov2l_study import DEFAULT_MANIFEST, load_rows, study_splits

    rows, _ = load_rows(DEFAULT_MANIFEST)
    splits = {k: v for k, v in study_splits(rows).items() if k != "future_all"}
    result = audit(splits)
    result["share_of_test_images"] = round(result["test_images_from_shared_sites"] / result["test_images"], 4)
    out = Path(__file__).resolve().parents[4] / "results/fmow_v2/site_overlap.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

