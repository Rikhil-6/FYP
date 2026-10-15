# ACDC paired-condition adaptation protocol

## Role of ACDC

ACDC is a realistic adverse-condition driving benchmark with fog, night, rain
and snow domains. Each adverse image is supplied with a normal-condition view
of the same scene, a 19-class semantic segmentation annotation and an invalid
region mask. This makes ACDC useful for testing whether an update learns
condition-invariant evidence rather than merely exploiting a synthetic
corruption.

ACDC is not a chronological stream. It therefore complements CLEAR rather than
replacing its temporal protocol. CLEAR tests deployment over time. ACDC tests
cross-condition adaptation on a safety-relevant dense prediction task.

The dataset requires registration and is licensed for non-commercial use.
Download `rgb_anon_trainvaltest.zip` and `gt_trainval.zip` from the
[official ACDC website](https://acdc.vision.ee.ethz.ch/download), then extract
them beneath `data/ACDC`. Build and validate the paired manifest with:

```bash
tools/uv run fyp-ingest-acdc --data-root data/ACDC
```

The scanner follows the layout used by the official CMA implementation. It
matches each adverse image to its `_rgb_ref_anon.png` normal reference,
`_gt_labelTrainIds.png` semantic label and `_gt_invIds.png` uncertain-region
mask. It fails if a required pair is incomplete.

## What counts as evidence of adaptation

Weight change alone is insufficient. A large parameter update can damage the
model, while a small targeted update can be useful. The evaluation therefore
combines four types of evidence.

1. **Task utility.** Report adverse-condition mIoU and macro F1 for fog, night,
   rain and snow. Report normal-condition scores and the clean-retention change.
2. **Condition invariance.** Compare normal and adverse representations with
   RBF MMD and a cross-validated domain-classifier ROC AUC after row-wise L2
   normalisation. A useful update should reduce both measures.
3. **Matched-scene consistency.** For the same-scene pairs, report cosine
   feature distance and prediction Jensen-Shannon divergence before and after
   adaptation. Use a paired bootstrap confidence interval for the reduction.
4. **Non-collapse and update tracing.** Report feature variance, effective rank,
   relative parameter change and the layers with the largest relative update.
   Lower MMD is not beneficial if the representation has collapsed or task
   performance has fallen.

The command below accepts historical and adapted feature archives. Each NPZ
file must contain paired `normal_features` and `adverse_features` matrices. It
may also contain `normal_probabilities`, `adverse_probabilities`,
`normal_saliency` and `adverse_saliency`.

```bash
tools/uv run fyp-audit-adaptation \
  --historical-features runs/acdc/historical_features.npz \
  --adapted-features runs/acdc/adapted_features.npz \
  --historical-checkpoint runs/acdc/historical.pt \
  --adapted-checkpoint runs/acdc/adapted.pt \
  --output runs/acdc/adaptation_evidence.json
```

The project can extract those archives directly from the historical and adapted
XAI ViT checkpoints:

```bash
tools/uv run fyp-screen-acdc \
  --manifest data/ACDC/paired_manifest.csv \
  --historical-checkpoint runs/xai/<historical-run>/models/best.pt \
  --adapted-checkpoint runs/siamese/<adapted-run>/models/best.pt \
  --output runs/acdc/paired-screen
```

This command is deliberately labelled a representation screen. The checkpoints
retain the CLEAR-100 classification head, whose labels have no valid mapping to
ACDC's semantic classes. ACDC task-performance claims require a 19-class ViT
segmentation head and a matched normal-to-adverse source training protocol.

Spatial saliency comparison is disabled by default. The normal and adverse
frames are GPS-matched but are not pixel-aligned. Patch maps should only be
compared after correspondence warping and invalid-region masking, as in
[Contrastive Model Adaptation](https://github.com/brdav/cma). Enable the check
with `--saliency-aligned` only after that preprocessing has been applied.

## Decision rule

An adaptation claim is supported when the adapted model improves adverse mIoU,
retains normal-condition performance, reduces MMD and domain separability, and
reduces matched-pair feature distance with a confidence interval above zero.
Feature variance and effective rank must remain stable enough to rule out a
trivial constant representation. Disagreement among these measures is reported
as mixed evidence rather than compressed into a single favourable score.

This protocol follows the paired normal/adverse rationale of the
[ACDC benchmark](https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html)
and the condition-invariant feature objective of CMA. MMD and the domain
classifier remain external diagnostics: they are not training losses in the
current model and therefore provide a separate check on the claimed behaviour.
