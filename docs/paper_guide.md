# Paper guide

This guide is for someone who has read (or is about to read) the paper and wants to know **what each part does,
where the code is, and how to rerun it**. All commands assume `source scripts/fmow_v2/env.sh` has been run with
paths edited for your machine (see [getting_started.md](getting_started.md)); `$FYP_RUNS` is where run outputs go.

## The idea in one paragraph

A linear classifier on a frozen DINOv2 Vision Transformer scores an image as the *average* of per-patch scores,
so the score each patch gives to the predicted class is an exact measure of how much it contributed. This
per-patch score, standardised within the image, is the **saliency map**. The study uses it to guide an update of
the model on newer data and compares: learned attention (no saliency), a hard and a soft saliency gate, LoRA
(which changes the backbone's features) and a hybrid of LoRA and a gate. Every saliency model has a **shuffled
twin** (same saliency values in random positions) so that the effect of the map's content is separated from the
effect of extra parameters.

## Section by section

| Paper section | What it does | Code | Command / output |
|---|---|---|---|
| 3 System design | Defines the historical head, the saliency map, the heads (learned attention, hard gate, soft gate), LoRA and the hybrid | `evaluation/saliency_attention_study.py` (heads), `soft_saliency_heads.py` (soft gate), `fmow_v2/equal_lora.py` (LoRA), `fmow_v2/hybrid.py` (hybrid) | see steps 4 and 9 below |
| 4.1 Data | Country-matched 42k-image subset, location-grouped splits, 2x crop around the labelled object | `fmow_v2/download_subset.py`, `evaluation/fmow_dinov2l_study.py` | `python -m fyp_drift.evaluation.fmow_v2.download_subset`; [data.md](data.md) |
| 4.2, 5.1 Drift and harm | Gaussian-kernel MMD, domain-classifier AUC, accuracy gap with location-clustered bootstrap, cloud calibration (7.5-point drop) | `evaluation/fmow_dinov2l_study.py --stages gate calib` | `$FYP_RUNS/main/` |
| 4.2, 5.4 Clouds | Training overlay, held-out cumulus (beside / over the object), real clouds (cover >= 10%), class-matched real-cloud comparison | `fmow_dinov2l_study.py` (overlay, cumulus), `fmow_v2/tokens.py` (token caches), `fmow_v2/control_hist.py` (class-matched numbers) | `fmow_v2.tokens cloud`, `fmow_v2.tokens heldout` |
| 5.4 Graded exposure (Table 6) | Mixes each clean-trained model's clean and overlay predictions on 25 / 50 / 75% of test images | `fmow_v2/graded_cloud.py` | `python -m fyp_drift.evaluation.fmow_v2.graded_cloud` -> `$FYP_RUNS/control_hist/mix.json` |
| 5.2 Head vs recent data | Trains learned attention, hard gate and twin on the 2010-13 data to separate the effect of the head from that of recent images | `fmow_v2/control_hist.py` | `python -m fyp_drift.evaluation.fmow_v2.control_hist` -> `control_hist/` |
| 5.2, 5.3 Accuracy and cost | Tables 4 and 5, Figures 5-7: macro-F1 under every condition, parameters, weights changed, training time | all of the above plus `fmow_v2/analysis.py`, `fmow_v2/summarise_extra.py` | `python -m fyp_drift.evaluation.fmow_v2.summarise_extra` (3-seed means) |
| 5.5 Hybrid | LoRA in the last two blocks plus a width-8 soft-gated scorer; rank 1 and rank 2; saliency-free control with `--kind attention` | `fmow_v2/hybrid.py` | `python -m fyp_drift.evaluation.fmow_v2.hybrid --rank 2 --blocks 2 --seed 42 [--kind attention]` |
| 6.1 Centre bias, moved objects | Re-crops test images so the object is off-centre; compares weight on the box against a fixed centre map | `fmow_v2/shift_test.py` | `$FYP_RUNS/shift_test/` |
| 6.2 Hard vs soft | Learned gate parameters (soft-gate floor rho) | `soft_saliency_heads.py` | `saliency_soft/` |
| 6.4 Faithfulness, stability, calibration | Uniform-swap flips, deletion tests, rank correlation across seeds, ECE | `evaluation/extra_metrics.py`, `fmow_v2/analysis.py` | `extra_metrics/`, `analysis/` |
| 6.5 Exact explanations | Per-patch contributions of the old and updated head (reconstruction error ~1e-5) | `fmow_v2/explain.py` | `python -m fyp_drift.evaluation.fmow_v2.explain` -> `analysis/explain_examples.npz` |
| Appendix C.1, C.2 | Saliency used only during training (guided head, masked-consistency, focus and context augmentation) | `fmow_v2/saliency_learning.py` | `saliency_learning/` |
| Appendix C.3 | Router that chooses between the old and the updated head per image | `evaluation/saliency_router_study.py` | `saliency_router/` |
| Appendix C.5 | The same attention-weight and before/after figures on cloudy images | `fmow_v2/cloud_examples.py`, `figures/fmow_v2/plots/fig_cloud_examples.py` | `python -m fyp_drift.evaluation.fmow_v2.cloud_examples`, then `python fig_cloud_examples.py cloud_examples.npz OUTDIR` |

## Rerunning the numbers

```bash
# one-off set-up
pip install -e . && pip install seaborn scipy safetensors matplotlib
source scripts/fmow_v2/env.sh

# 1. data and token caches (see fmow_v2.md for the full order)
python -m fyp_drift.evaluation.fmow_v2.download_subset
python -m fyp_drift.evaluation.fmow_v2.tokens cloud

# 2. heads (about 1 minute each once tokens exist)
python -m fyp_drift.evaluation.saliency_attention_study --run $FYP_RUNS/main
python -m fyp_drift.evaluation.soft_saliency_heads

# 3. LoRA and hybrid (about 30 minutes each on a T4); seeds 42, 43, 44
bash scripts/fmow_v2/run_equal_training.sh
for s in 42 43 44; do python -m fyp_drift.evaluation.fmow_v2.hybrid --rank 2 --blocks 2 --seed $s; done

# 4. controls and graded exposure
python -m fyp_drift.evaluation.fmow_v2.control_hist
python -m fyp_drift.evaluation.fmow_v2.graded_cloud
python -m fyp_drift.evaluation.fmow_v2.summarise_extra

# 5. figures
bash figures/fmow_v2/plots/make_all.sh
```

Every run writes one JSON per model, condition and seed with per-image predictions, so any number in the paper
can be recomputed from `results/fmow_v2/run_outputs/` without retraining.

## How to read the key quantities

- **Macro-F1** is computed over all 62 classes (`labels=np.arange(62)`), so a class the model never predicts
  counts as zero. Using sklearn's default (only classes present) gives about 0.8 points more.
- **Real-cloud drop** is reported against a *class-matched* clean set: cloud-free test images weighted to the
  class mix of the 920 cloudy images, because the cloudy images are not a random sample of classes.
- **Trainable parameters** and **backbone weights changed** are different costs. Rank-2 LoRA on six blocks
  trains 221k parameters but changes about 75.5 million weights (a quarter of the backbone), because the
  low-rank update covers whole matrices. The heads train a similar number and change none.
- **Shuffled twin**: same model, same saliency values, random positions. If it performs as well, the benefit did
  not come from where the evidence lies.

## Known limits

One dataset (fMoW), one frozen backbone (DINOv2-L/14), saliency taken from the historical head itself, cloud
injected as a designed scenario (natural cloud cover does not rise over 2010-17 in this data), and the
natural-drift harm test allows up to 2.1 points of harm. See Section 7.3 of the paper.


## Audit checks added after review

* Cross-period sites: splits are grouped by country and UTM zone within each period, which does not stop a site recurring in a later period. `python -m fyp_drift.evaluation.fmow_v2.site_overlap` measures it (`results/fmow_v2/site_overlap.json`: 334 of the 7,077 test images, 4.7%, come from a site that also occurs in an earlier split); `results_table.json` also reports every model's clean macro-F1 on the remaining test images.
* One results table: `python -m fyp_drift.evaluation.fmow_v2.aggregate` rebuilds every table row from the raw per-seed runs; the result figures read it.
* Tests: `PYTHONPATH=src pytest tests/test_fmow_v2_results.py` checks the aggregation, that the published tables match the raw runs, the figure inputs, that every saved model loads, and the site audit.
