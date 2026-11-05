# Data manifests

The exact fMoW-RGB subset used in the paper: 41,969 images (about 14,000 per period, the same country quotas in every period; 31 images that
failed to download in the original run are not listed).

| File | Columns |
|---|---|
| `fmow_subset_manifest.csv.gz` | `key` (path in the public bucket), `split`, `category`, `timestamp`, `country`, `utm`, `period`, `size_bytes` |
| `fmow_subset_acquisition_metadata.csv.gz` | `key`, `cloud_cover`, `off_nadir_angle`, `sun_elevation`, `sun_azimuth`, `target_azimuth`, `gsd`, object box (`box_cx`, `box_cy`, `box_w`, `box_h`) |

`python -m fyp_drift.evaluation.fmow_v2.download_subset` downloads the images from
`https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-rgb/` and rebuilds the two CSV files the experiments read.
The row order is the order used in the paper (it affects the grouped splits).
