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
