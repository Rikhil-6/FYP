# FMoW saliency guided adaptation study

## Question

Can patch evidence improve supervised adaptation under visual covariate drift beyond ordinary adaptation and matched map controls?

The proposed correction mechanism is explicit: use labels from the adaptation window to update a small part of the backbone, while the frozen historical model's saliency map identifies patch representations that the update should preserve. This may help when new acquisition conditions disturb the image but the scene label remains valid. If the saliency map is informative, the adapted model should recover task performance with less damage to useful historical evidence than the matched uniform or shuffled anchors. The project tests this claim; it does not assume it is true.

Low classification accuracy alone does not establish drift. First compare cluster-disjoint historical out-of-fold performance with future performance and report the clustered confidence interval for the macro-F1 drop. Report class-prior change separately. A feature MMD difference establishes a representation change, not task harm. If natural task harm is not established, the natural-data outcome is an adaptation check; the calibrated localized shift becomes the controlled test of drift mitigation.

## Shared model and arms

The historical DINOv2-small backbone and patch-mean linear head are the common starting point. The head stays frozen during adaptation. Each trainable arm uses the same E, M, or L four-block LoRA group, labeled 2014–15 train split, cluster-disjoint 2014–15 validation split, future test split, seed, learning rate, batch size, and number of epochs. Training batches are deterministically shuffled **within** the adaptation split. The temporal split is unchanged.

- `frozen`: historical model without adaptation.
- `plain`: supervised cross-entropy, no anchor (λ = 0).
- `uniform`: cross-entropy plus uniform patch-token anchoring to the frozen backbone.
- `shuffled`: same anchor weight and top-40% area as saliency, with patch locations permuted.
- `saliency`: same anchor weight, using the frozen head's top-40% predicted-class patch evidence.

The anchor is mean squared change in final patch tokens, spatially weighted by the selected map. The map is computed from the frozen reference model for each adaptation image. The head's patch evidence sums to its logit after the bias, but does not explain the backbone's entire decision process. All four trainable arms update the same number of parameters.

The primary endpoint is 62-class macro-F1 on cluster-disjoint `future_test`. Also save accuracy, balanced accuracy, classwise F1, prediction histograms, training cross-entropy, sample prediction entropy, marginal entropy, and wall-clock time. A saliency-specific claim requires saliency to beat **both** map controls with paired cluster-bootstrap intervals across seeds 42–44. The plain arm establishes whether supervised adaptation works without an anchor.

## Controlled shift

The `cloud` setting adds a deterministic, localized near-white overlay to images in the adaptation and future periods. Each image path and seed fixes the same mask across arms. If `--cloud-alpha` is omitted, the script checks candidate strengths on **historical validation images only** and chooses one causing a 5–10 macro-F1 point frozen-model drop; it stops if none qualifies. Future labels are never used to choose severity. The controlled images require human review for label preservation before a real-world claim.

## Run

Use the same saved historical head and manifest for each seed. Each output directory must be unique to its seed, shift, and block group. Existing arm JSON files are skipped unless `--force` is given.

```bash
PYTHONPATH=src python -m fyp_drift.evaluation.fmow_study_b \
  --manifest data/fmow_v1.1/subset_7k_country_matched/manifest_downloaded.csv \
  --head runs/historical-head/head.pt \
  --output runs/study-b-clean-seed42 --seed 42 --shift clean --group E
```

For the controlled shift, change the output directory and use `--shift cloud`. Run seeds 42, 43, and 44. Use historical validation to fix the head and hyperparameters before reading future-test outcomes. The historical harm gate, yearly windows, and explanation-map validity checks remain separate required studies; this command does not implement them.
