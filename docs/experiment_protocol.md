# Experiment protocol: source-separated acquisition and context shifts

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
| `realistic_mix` | Several changes in the same deployment period | 1/2/3 non-occlusion sources; occlusion added to 10% of samples |

The final mixed condition logs its component names for every sample. This lets a
reviewer group failures by actual source rather than infer drift from the image.

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
  --permutations 199
```

No single value proves harmful drift. MMD tests distribution equality and the
domain classifier measures how reliably held-out features reveal their source.
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
  --download
```

FMoW is the primary external dataset because its satellite imagery includes
explicit temporal and geographical shift. The probe is fitted on `train`,
measured on `id_val`, and then measured on the out-of-distribution `val` split.
MMD and domain-classifier ROC AUC compare the two validation feature samples.
The change in accuracy and macro F1 determines whether the detected shift is
harmful to the probe.

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
- Confirm `realistic_mix` uses several compositions and occlusion remains near
  its declared 10% rate.
- Train the realistic profile and a matched control without auxiliary saliency
  losses using the same seed, epochs, data, and trainable blocks.
- Report each individual source, the mixed condition, clean performance, helpful
  and harmful routes, and component-stratified results.
