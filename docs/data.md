# Data

**Source:** fMoW-RGB, public on AWS at `s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/` (HTTPS mirror
`https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-rgb/`). Each image comes with a JSON file
holding the object box, timestamp, sun angles, off-nadir angle and cloud cover.

**Subset.** `fmow_v2.expand_subset` picks about 14,000 images for each period with the same per-country quota in
every period, so geography does not change between periods. 31 of about 42,000 downloads failed.

| Period | Role | Split (by location) | Images |
|---|---|---|---|
| 2010–13 | Train the historical model | hist_train / hist_val | 11,790 / 2,193 |
| 2014–15 | Adapt (update) the model | adapt_train / adapt_val | 12,572 / 1,420 |
| 2016–17 | Test | future_test (half of the locations) | 7,077 |

Splits are by location group (country + UTM zone), so no site appears in two splits.

**Preparation.** Each image is cropped to a square twice the object box's longer side (at least a quarter, at
most all of the shorter side) and resized to 280 × 280. Crops are cached once as a uint8 memmap
(`$FYP_CACHE`), and frozen DINOv2-L tokens are cached per split (`$FYP_TOKENS`, pooled to 10 × 10 cells).

**Clouds.** Training overlay: near-white cloud on the brightest 20% of a smoothed noise field (α = 0.85).
Held-out test clouds: cumulus with sun-consistent shadows, beside or over the object. Real clouds: test images
whose metadata cloud cover is at least 10%.
