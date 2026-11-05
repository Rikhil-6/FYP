# Getting started

## 1. Install

```bash
git clone https://github.com/Rikhil-6/FYP.git && cd FYP
python -m venv .venv && source .venv/bin/activate     # Python 3.11
pip install -e .                                       # installs fyp_drift and its dependencies
pip install seaborn scipy safetensors                  # figures and saved models
```

A GPU is needed for training (a 16 GB Tesla T4 was used). Inference with a saved model runs on CPU.

## 2. Get the data

The experiments use **fMoW-RGB** (Functional Map of the World), public on AWS
(`s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/`, no account needed). The exact 41,969-image subset is listed in
`data_manifests/`, so nothing else is required:

```bash
export FYP_ROOT=$HOME/fyp          # any folder with ~40 GB free; every path derives from it (see scripts/fmow_v2/env.sh)
source scripts/fmow_v2/env.sh
python -m fyp_drift.evaluation.fmow_v2.download_subset --limit 50   # quick test first, then run without --limit
```

`download_subset` fetches the listed images over HTTPS, checks each against its published byte size and writes
`manifest_downloaded.csv` and `acquisition_metadata.csv` (cloud cover, sun angle, off-nadir, object box) under
`$FYP_DATA/subset_14k_country_matched/`, which is where `FYP_MANIFEST` and `FYP_METADATA` point. The images are identical to
the ones used in the paper. `fmow_v2.expand_subset` is the original tool that grew the subset from a 7k pilot and is only needed to
re-draw the subset from the full fMoW metadata. See [data.md](data.md) for the periods, splits and file layout.

## 3. Run the experiments

The full order is in [fmow_v2.md](fmow_v2.md). The short version:

```bash
python -m fyp_drift.evaluation.fmow_dinov2l_study --output $FYP_RUNS/main      # drift, harm gate, cloud calibration
python -m fyp_drift.evaluation.saliency_attention_study --run $FYP_RUNS/main   # attention and saliency heads
bash scripts/fmow_v2/run_equal_training.sh                                       # LoRA and the hybrid, 3 seeds
```

## 4. Rebuild the figures

```bash
bash figures/fmow_v2/plots/make_all.sh
```

## 5. Use a trained model

```bash
python -m fyp_drift.evaluation.fmow_v2.models predict models/fmow_v2/soft_gate my_site.jpg --map attention.png
```

See [models.md](models.md).
