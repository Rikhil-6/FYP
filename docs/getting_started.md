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

The experiments use **fMoW-RGB** (Functional Map of the World), which is public on AWS
(`s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/`, no account needed).

```bash
source scripts/fmow_v2/env.sh                          # edit the paths at the top for your machine
python -m fyp_drift.evaluation.fmow_v2.expand_subset   # country-matched subset, ~14k images per period
python -m fyp_drift.data.fmow_metadata                 # per-image sun angle, cloud cover, off-nadir
```

`expand_subset` downloads only the images in the subset (about 42k JPEGs) plus their JSON metadata and writes
`manifest_downloaded.csv`. See [data.md](data.md) for the periods, splits and file layout.

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
python -m fyp_drift.evaluation.fmow_v2.predict models/fmow_v2/soft_gate my_site.jpg --map attention.png
```

See [models.md](models.md).
