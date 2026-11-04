# fMoW v2 experiments

Code for the paper *Saliency-Guided Attention on a Frozen Vision Transformer for Temporal Adaptation under
Cloud Cover*. The base study lives in `src/fyp_drift/evaluation/` (`fmow_dinov2l_study.py`,
`saliency_attention_study.py`, `soft_saliency_heads.py`, `saliency_router_study.py`, ...). The v2 additions
are in `src/fyp_drift/evaluation/fmow_v2/`.

## Setup

```bash
source scripts/fmow_v2/env.sh   # paths for data, caches and outputs (edit for your machine)
```

## Run order

| Step | Command | Output |
|---|---|---|
| 1. Build the 14k-per-period subset | `python -m fyp_drift.evaluation.fmow_v2.expand_subset` | manifest + images |
| 2. Drift, harm gate, cloud calibration, historical head | `python -m fyp_drift.evaluation.fmow_dinov2l_study --output $FYP_RUNS/main --stages gate calib` | `main/` |
| 3. Token caches (clean, then cloud) and 2x2 pooling | `saliency_adapters` (clean), `fmow_v2.tokens cloud`, `fmow_v2.tokens pool` | `$FYP_TOKENS` |
| 4. Attention and saliency heads | `saliency_attention_study`, `soft_saliency_heads` | `saliency_attention/`, `saliency_soft/` |
| 5. Parameter sweep and cloud-aware pooling | `fmow_v2.cloud_pooling` | `cloud_param/` |
| 6. Moved-object test, saliency in training | `fmow_v2.shift_test`, `fmow_v2.saliency_learning` | `shift_test/`, `saliency_learning/` |
| 7. Router | `saliency_router_study` | `saliency_router/` |
| 8. Held-out cloud tokens | `fmow_v2.tokens heldout` | `$FYP_TOKENS_EXTRA` |
| 9. Equal-training LoRA and hybrid | `scripts/fmow_v2/run_equal_training.sh` | `lora_fair/`, `hybrid/` |
| 10. Analysis and explanations | `fmow_v2.analysis`, `fmow_v2.explain` | `analysis/` |

Heads on held-out clouds reuse `saliency_attention_study --tokens DIR`, where `DIR` links the clean training
caches and the held-out test cache under the expected names.

## Protocol notes

- **Equal training.** Heads train for 15 epochs and LoRA/hybrid for up to 5, each keeping the best 2014–15
  validation epoch; LoRA stops after 2 epochs without improvement. LoRA's learning rate (3e-4) was chosen
  once on rank-2, clean, seed 42 from {1e-4, 3e-4}; heads use 1e-3.
- **LoRA sites.** q, k, v, attention output and both MLP layers of the last `--blocks` blocks
  (18,432 parameters per block per rank).
- **Losses.** The historical head uses class-weighted cross-entropy; all later heads and LoRA use plain
  cross-entropy.
- **Seeds.** Heads: 42, 43, 44. LoRA and hybrid: 42 first, 43 and 44 added afterwards.

## Outputs

- `results/fmow_v2/run_outputs/`: saved output of every run (per-image predictions); `results/fmow_v2/summary_tables.md` has the headline numbers.
- `figures/fmow_v2/`: every figure, the code that draws it (`plots/`) and its input data (`plots/data/`). Rebuild with `bash figures/fmow_v2/plots/make_all.sh`.
