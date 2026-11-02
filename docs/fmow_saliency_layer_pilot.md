# Saliency-guided layer-selection pilot

This is an exploratory architecture test on the custom FMoW-RGB chronological
subset. It does not replace the full drift protocol or establish covariate drift.

## Question

Can a frozen historical model's patch evidence guide **where** to adapt under
an unlabeled target-window shift, improving future macro-F1 over a feature-only
choice and a shuffled-evidence choice?

## Model and data flow

- DINOv2-small provides 12 blocks of patch tokens. A 62-class linear head is
  fitted to the mean of final patch tokens from 2010-13 `hist_train`, with
  cluster-disjoint `hist_val` for stopping. The head remains frozen afterward.
- The historical model's predicted-class patch contribution is
  `W[predicted_class] dot final_patch_token / patch_count`. Its sum plus bias
  must reconstruct that class's logit. The map explains the linear head's
  output; it is not an explanation of every backbone operation.
- Historical training images and unlabeled `future_a` images yield per-block
  patch tokens. Top-40% evidence weights pool each block's tokens. RBF MMD²
  compares historical and target pooled features. The E, M, or L group with
  the highest mean block MMD² is selected before adaptation or test scoring.
- Uniform patch weights and spatially shuffled, equal-area evidence weights
  produce separate control selections using the same images and MMD code.
- E (blocks 1-4), M (5-8), L (9-12), and ALL are trained separately from the
  same frozen model and head. Adapters update attention query/key/value,
  attention projection, and MLP fc1/fc2; the chosen blocks' LayerNorm affine
  parameters also update. E, M, and L must have equal trainable parameter
  counts. The target loss is prediction entropy minus batch-mean prediction
  entropy, using `future_a` without its labels. A0 is frozen inference.

## Readout

The primary pilot score is macro-F1 on cluster-disjoint `future_test`, over all
62 classes. Record accuracy and balanced accuracy too. The diagnostic is
useful only if its chosen group's score beats the uniform and shuffled choices
across meaningful replicated runs. If all selectors choose the same group,
this run cannot isolate a saliency-specific benefit. The best E/M/L arm is
reported after evaluation as an oracle comparison, never used to choose the
diagnostic. No hyperparameter or group choice uses `future_test` labels.

The 128-image-per-period server smoke test checks execution only. Its scores
are not evidence for the research question. A full run needs seeds 42, 43,
and 44, clustered uncertainty, natural-drift harm assessment, and controlled
shifts before any production-oriented conclusion.

Run one exploratory full seed with:

```bash
PYTHONPATH=src python -m fyp_drift.evaluation.fmow_saliency_layer_pilot \
  --manifest data/fmow_v1.1/subset_7k_country_matched/manifest_downloaded.csv \
  --output runs/fmow-saliency-layer-seed42 --seed 42
```
