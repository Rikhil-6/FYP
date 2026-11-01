# Experiment protocol: source-separated acquisition and context shifts

## Three separate experiments

1. **Unlabelled drift detection.** Given a limited sample of incoming raw
   images, embed them with the classifier's frozen, externally pretrained
   DINOv2 backbone and ask
   whether their distribution differs from a historical sample. MMD with a
   permutation test is the primary detector; grouped domain-classifier AUC is
   a complementary detector motivated by classifier two-sample tests. Report
   detection and effect-size estimates at fixed sample counts, including a
   clean split-half null. The observed failure to detect clean natural-year
   change with 300 samples is a power/representation boundary, not proof that
   no drift exists. Neither detector uses class labels or establishes task harm.
2. **Saliency-guided updating.** On the fixed chronology, compare the
   conventional uniform-pooling model, learned patch weighting without the
   auxiliary objectives, and learned weighting with those objectives. Report
   clean and every source-specific future condition. A fixed-checkpoint
   intervention then replaces learned weights with uniform and permuted
   weights while holding the class token, backbone, and classifier fixed. This
   tests whether the trained prediction depends on the map. A separately
   trained `patch_only` pooling variant removes the direct class-token-to-head
   bypass for both uniform and saliency arms. Both updated arms start from the
   same historical patch-only uniform checkpoint; the saliency gate starts at
   exact uniform weights. This removes the different-historical-head objection
   from the new arm comparison. The backbone still retains transformer attention
   between class and patch tokens, so this is a correction of the direct head
   path, not proof that patch features exclude all class-token information.
   Neither intervention alone shows that the map causes a generalisation gain.
3. **Selective routing.** Given the frozen historical and updated saliency
   models from experiment 2, test whether the routing policy adds useful
   decision value beyond always-updated and confidence-only choices. Compare
   policies at matched route coverage, report harmful and missed switches,
   and state the cost or risk target before making a safety claim. Routing is
   a separate decision experiment, not the measure of saliency's effect on
   the updated classifier.

The completed corrected-split conventional arm scores 81.55% macro F1 on the
2014 evidence-ranked mixture, versus 80.78% for updated saliency and 80.81%
for saliency without auxiliary losses. The saliency update is below conventional
fine-tuning on nine of eleven 2014 conditions, tied on one, and above on one.
The archived 1.20-point saliency advantage therefore does not replicate in
this corrected single-seed strategy comparison. The heads begin from different
historical checkpoints; the queued shared-checkpoint `patch_only` factorial
is the required component-level test.

## Why the earlier combined condition was inadequate

The original `combined` transform applied saturation reduction, Gaussian noise,
and a square occlusion to every evaluated image. It was deterministic and useful
as a reproducible stress test, but its panels visually overemphasized missing
pixels. It also mixed distinct causes into one score, making it difficult to say
whether a model handled sensor degradation, illumination, viewpoint, or content
loss. The old transform remains available for exact reproduction and is labelled
legacy in the code.

CLEAR already provides the primary natural temporal shift through images uploaded
from 2004 to 2014. The transforms below are controlled acquisition stressors on
top of that real chronology. They are not estimates of how frequently each event
occurs in a particular industry.

## Source-separated benchmark

| Scenario | Plausible source | Progressive severities |
|---|---|---|
| `clean` | Natural CLEAR temporal evolution only | No synthetic change |
| `illumination` | Exposure, lighting, time of day | ±6/12/18% brightness plus a smaller contrast change |
| `white_balance` | Camera or ISP colour temperature | ±4/8/12% red-blue channel gain |
| `sensor_noise` | Shot and read noise from sensor or low light | Signal-dependent mild noise |
| `defocus_blur` | Autofocus error or depth of field | 3/5/7 pixel Gaussian kernel |
| `motion_blur` | Camera or subject motion | 3/5/7 pixel directional kernel |
| `jpeg_compression` | Re-encoding or bandwidth limits | JPEG quality 85/70/55 |
| `resolution_loss` | Sensor change, downsampling, bandwidth | 192/160/128 pixel intermediate resolution |
| `viewpoint_shift` | Mounting, framing, crop, pose | Small translation, rotation, and scale change |
| `occlusion` | Temporary obstruction or missing content | 1.5/3/5% local mean patch |
| `realistic_mix` | Several changes in the same deployment period | Top 1/2/3 sources from the frozen 2011 evidence ranking |

The 2011 source screen ranks individual transformations by the mean of their
descending MMD-squared and domain-AUC ranks. The top three are motion blur,
sensor noise, and JPEG compression. The final mixed condition applies the top
one, two, or three sources as severity rises and logs those names for every
sample. The 2012-2014 evaluation labels do not influence this selection.

## Adaptation and frozen evaluation

The `realistic` adaptation profile retains the established chronology:

1. Historical model training: 2004–2008.
2. Historical checkpoint selection: 2009.
3. Newer-model adaptation: 2010.
4. Adapted-checkpoint selection and router calibration: disjoint, deterministic,
   class-stratified partitions of 2011 (50/50 by default).
5. Frozen evaluation: 2012–2014.

The 2010 and 2011 streams contain 25% clean examples and 75% examples with one
individual source. Each of the nine individual sources occupies 1/12 of the
stream, so occlusion is 8.3% rather than being present in every image. Compound
mixtures remain held out until final evaluation. The 2012, 2013, and 2014 tests
use severity levels one, two, and three respectively.

This partition prevents the same labels from selecting the adapted checkpoint
and fitting the route coefficients and thresholds. Earlier reported runs predate
this correction and must be identified as exploratory shared-calibration results.
Use `--reuse-calibration-for-selection` only to reproduce those archived runs.

The corrected conventional fine-tuning control uses a historical
uniform-pooling checkpoint trained on 2004–2008 and selected on 2009. It keeps
the saliency arm's 2010 realistic adaptation transform, five-epoch budget,
last-three-block tuning, optimiser rates, seed, and class-stratified 2011
checkpoint-selection partition. Its configuration now records both the
selection and held-out calibration counts and rejects a resume with a different
calibration protocol. `deploy/run_corrected_finetune_20261031.sh` trains this
control and evaluates the same 11 named conditions in 2012–2014. This is a
strategy comparison: the uniform and saliency heads differ in architecture and
historical checkpoint, so it does not isolate a saliency mechanism by itself.

## When saliency guidance is expected to help

The strongest rationale is class-conditional covariate drift. The label meaning
and relevant object evidence remain stable, but background, capture conditions,
or low-level appearance change. A saliency-gated model may then reduce reliance
on a historical contextual shortcut and preserve class evidence from stable
patches. This is especially plausible for framing, localized obstruction, and
background or viewpoint change.

The rationale is weaker for uniform global degradations such as strong blur or
JPEG compression. Saliency cannot recover visual information that has been lost;
any gain there may come from ordinary robustness training rather than the
explanation mechanism. Results must therefore be reported by source, with a
matched no-auxiliary-saliency control, rather than only as one mixed score.

The method is not intended to solve label-definition drift, new classes, severe
object disappearance, or annotation errors. A faithful saliency map can still
explain a prediction made under the wrong task definition.

## Router evidence ablation

The router can be calibrated with prediction evidence, explanation evidence, or
both. This supports a matched test of whether saliency contributes beyond changes
in confidence and predictive entropy:

```bash
tools/uv run fyp-train-router ... --router-features confidence_only
tools/uv run fyp-train-router ... --router-features saliency_only
tools/uv run fyp-train-router ... --router-features combined
```

Prediction-only calibration omits the saliency eligibility floor. Saliency-only
and combined calibration retain it, and every saved policy records the features
used to fit its coefficients.

For a matched feature test after a corrected long-horizon training run has
finished, fit all three policies against the same frozen model pair and 2011
calibration images without retraining:

```bash
tools/uv run python -m fyp_drift.evaluation.router_feature_ablation \
  --run-dir runs/siamese/<corrected-run> \
  --output runs/feature-ablations/<new-id>
```

This command requires the source run's `logs/drift_routing_decisions.csv` and
refuses a legacy shared-calibration run. It compares the native fitted policies
and ranks disagreeing predictions at fixed 10%, 25%, and 50% route coverage.
The latter comparison uses the same number of routes in every feature set and
does not tune thresholds on future labels. It is a transductive diagnostic: it
uses the full unlabelled test group to set a route quota and is not a streaming
deployment policy. The logged future features are rounded to six decimal
places. The command requires those values to reproduce every original policy
decision; if rounding changes one, evaluate the future images again with
full-precision features. The refitted policy must also match the source policy
on the full-precision calibration rows.

For the completed corrected fixed-model comparison, the retrospective
decision-cost summary is in `results/validation-20261031/routing_utility.json`.
It uses beneficial replacements minus lambda times harmful replacements,
relative to always-historical use. On the 2014 evidence-ranked mixture, the
available pair has 379 beneficial and 198 harmful replacements. At lambda=1,
always-updated leads; at lambda=2, confidence-only leads; at lambda=3,
always-historical leads. The combined saliency-and-confidence gate is not the
best option at any of those costs, including at matched 50% disagreement
coverage. Lambda is hypothetical and selected for sensitivity analysis after
seeing the results, so this is not a deployment utility estimate.

## Independent drift verification

Classification loss does not establish that two input distributions differ.
The drift verification command therefore embeds images with the frozen
DINOv2-small backbone and compares distributions before using model accuracy to
judge harm. It reports two complementary quantities:

- RBF maximum mean discrepancy squared, with a permutation p-value;
- cross-validated domain-classifier ROC AUC.

The 2014 source tests pair every shifted image with its clean counterpart. This
controls image identity and class composition when verifying motion blur,
occlusion, and the other acquisition changes. A clean 2014 split-half comparison
provides a within-distribution reference. Natural temporal comparisons use clean
2009 images as the historical reference and clean 2012-2014 images as future
samples. Those comparisons capture temporal change but may also reflect changes
in class frequency and image composition, so they are reported separately.

Run the verification before the model comparison:

```bash
tools/uv run fyp-verify-drift \
  --data-root data \
  --output runs/drift-verification \
  --limit 500 \
  --permutations 499
```

No single value proves harmful drift. MMD tests distribution equality and the
domain classifier measures how reliably held-out features reveal their source.
The saved AUC is the raw out-of-fold value for the assigned domain labels.
Values below 0.5 indicate reverse ranking under that score direction; they
must not be described as symmetrised evidence of no drift.
Performance loss and routing outcomes determine whether an observed shift
matters to the task.

MMD follows the kernel two-sample test of Gretton et al. The domain detector
follows the classifier two-sample test of Lopez-Paz and Oquab. Both methods have
independent open-source implementations in Alibi Detect as `MMDDrift` and
`ClassifierDrift`, which provides a public reference for the test design:

- <https://www.jmlr.org/papers/v13/gretton12a.html>
- <https://arxiv.org/abs/1610.06545>
- <https://github.com/SeldonIO/alibi-detect>

## Cross-dataset replication screen

Run the same feature-space tests and a frozen-DINOv2 linear probe on official
WILDS splits:

```bash
tools/uv run fyp-replicate-wilds \
  --dataset fmow \
  --data-root data/wilds \
  --output runs/wilds/fmow \
  --architecture-screen \
  --download
```

FMoW is the primary external dataset because its satellite imagery includes
explicit temporal and geographical shift. The probe is fitted on `train`,
measured on `id_val`, and then measured on the out-of-distribution `val` split.
MMD and domain-classifier ROC AUC compare the two validation feature samples.
The change in accuracy and macro F1 determines whether the detected shift is
harmful to the probe.

With `--architecture-screen`, the command also trains the baseline and
saliency-gated classifier heads over the same frozen DINOv2 backbone. Both see
the same sampled training images, epochs, seed, and optimiser settings. Their
ID and OOD macro F1 scores test whether the modified classifier changes
generalisation under the dataset's native shift. The screen freezes the
backbone to keep multi-dataset runs practical, so its result must be kept
separate from the partially fine-tuned CLEAR experiment.

iWildCam can test camera-location shift and Camelyon17 can test hospital shift
with the same command. They broaden the evidence but do not verify temporal
replication. A full claim about router transfer requires matched training,
calibration, seeds, and saliency audits on the chosen dataset; this lightweight
screen deliberately makes the narrower claim that the detection and
performance-evaluation protocol can be repeated.

## Evidence base

- Hendrycks and Dietterich group common corruptions into noise, blur, weather,
  and digital categories in ImageNet-C: <https://arxiv.org/abs/1903.12261>.
- CLEAR supplies smooth natural temporal evolution and near-future streaming
  evaluation rather than artificial year-by-year IID splits:
  <https://arxiv.org/abs/2201.06289>.
- WILDS documents real shifts across camera locations, geography, institutions,
  and time: <https://arxiv.org/abs/2012.07421>.
- Beery et al. show poor recognition generalization across new camera-trap
  locations where backgrounds and viewpoints change:
  <https://openaccess.thecvf.com/content_ECCV_2018/html/Beery_Recognition_in_Terra_ECCV_2018_paper.html>.
- ImageNet-ES studies environmental and camera-sensor shifts captured with a
  physical camera, illustrating the gap between digital corruption and actual
  acquisition changes:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Baek_Unexplored_Faces_of_Robustness_and_Out-of-Distribution_Covariate_Shifts_in_Environment_CVPR_2024_paper.html>.
- Zheng et al. motivate saliency-guided adaptation when background variation
  dominates domain change: <https://arxiv.org/abs/2403.12712>.

## Pre-run checks

- Confirm every transform is deterministic for a fixed seed and manifest index.
- Save a visual grid with clean and every severity-three source before training.
- Confirm `realistic_mix` uses the frozen 2011 top-three source ranking.
- Train the realistic profile and a matched control without auxiliary saliency
  losses using the same seed, epochs, data, and trainable blocks.
- Report each individual source, the mixed condition, clean performance, helpful
  and harmful routes, and component-stratified results.
