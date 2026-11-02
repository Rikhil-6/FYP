# Drift validation and external replication

## What is being verified

The verification step asks whether a reference image sample and a later or
corrupted sample produce different frozen DINOv2 feature distributions. It uses
two tests whose outputs can be checked independently:

1. **Maximum mean discrepancy (MMD).** An RBF-kernel MMD squared value records
   the observed separation. A permutation test estimates the probability of a
   separation at least this large if the samples were exchangeable.
2. **Classifier two-sample test.** A balanced logistic classifier is trained and
   evaluated by five-fold cross-validation to distinguish reference from target
   features. The saved AUC is the raw out-of-fold ROC AUC for the assigned
   domain labels; it is not symmetrised. A value of 0.5 is chance ranking,
   values above 0.5 rank the target domain higher, and values below 0.5 rank
   it lower. Interpret departures in either direction with sampling uncertainty.

The code is intentionally limited to these two measures. Gretton et al. define
the kernel two-sample test, while Lopez-Paz and Oquab describe the classifier
two-sample test. Alibi Detect provides public `MMDDrift` and `ClassifierDrift`
implementations against which the methodology can be inspected.

- Gretton et al.: <https://www.jmlr.org/papers/v13/gretton12a.html>
- Lopez-Paz and Oquab: <https://arxiv.org/abs/1610.06545>
- Alibi Detect source: <https://github.com/SeldonIO/alibi-detect>
- Alibi Detect drift documentation: <https://docs.seldon.ai/alibi-detect/cd>

## CLEAR controls

Three comparisons prevent a significant result from being treated as sufficient
evidence on its own:

- a seeded random clean split-half control estimates within-distribution
  variation without retaining dataset order;
- clean historical and future years measure natural temporal separation;
- clean and shifted versions of the same 2014 images isolate each controlled
  corruption from image identity and class composition.

The paired design is part of sampling, even though MMD and the domain classifier
remain distribution-level tests. A corruption is measurable when its MMD is
larger than the null control, its permutation p-value is small, and its domain
ROC AUC rises above chance. It is harmful only when classification performance
also falls. This distinction avoids presenting any visible change as harmful
data drift.

### Unlabelled sample-size sensitivity

The earlier 300-image audit balanced samples by class retrospectively. Its MMD
estimate used all 300 images per group, but each permutation test subsampled
200 per group. The new `fyp_drift.evaluation.drift_sample_size` audit selects
images randomly within year without class labels and uses the full sampled
group for each of 499 permutations. It tests nested samples from one seed, so
the rows are related rather than independent replications.

| Images per group | Clean 2009 vs 2014 MMD p | Clean 2014 null p | Motion blur p | Evidence-ranked mix p |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.268 | 0.898 | 0.002 | 0.002 |
| 300 | 0.058 | 0.514 | 0.002 | 0.002 |
| 500 | 0.136 | 0.678 | 0.002 | 0.002 |

Motion-blur AUC is 0.989–0.994 and mixture AUC is 0.999–1.000; the clean-year
AUC range is 0.457–0.565. The controlled changes are detectable here even with
100 images per group. The clean-year non-detection through 500 images is a
limit of this representation, draw, and detector setup, not evidence that
natural drift is absent. Random sampling may also vary the class mixture.
The complete record is in
`results/validation-20261031/drift-sample-size-100-300-500.json`.

## WILDS replication

`fyp-replicate-wilds` repeats the detector and performance analysis on standard
WILDS splits. It fits a linear classifier to frozen DINOv2 training features and
reports accuracy and macro F1 on in-distribution and out-of-distribution
validation data.

The optional `--architecture-screen` adds a matched comparison between the
uniform-patch baseline and the learned saliency gate. It trains the task heads
over the same frozen backbone, images, seed, epochs, and optimiser. This provides
a direct, affordable test of the modified classifier across datasets while
avoiding the false claim that a CLEAR-calibrated router transfers unchanged.

FMoW is the first choice because it is an image classification dataset with
time and geography in its distribution shift. iWildCam is a useful secondary
test of camera-location shift. Camelyon17 tests hospital acquisition shift, but
it is not temporal. The WILDS repository supplies the dataset loaders and split
definitions: <https://github.com/p-lambda/wilds>.

The resulting JSON separates two claims:

- MMD and domain ROC AUC show whether the frozen feature distribution changed;
- the ID-to-OOD accuracy and macro-F1 differences show whether that change
  reduced predictive performance.

This screen does not by itself reproduce the saliency-guided router. A full
architecture replication must retrain and calibrate the historical model,
updated model, and router on each dataset with matched seeds and budgets.

Wild-Time Yearbook is included as a small, downloadable temporal check when the
larger WILDS archives are unavailable. It trains on 1930-1989, evaluates an ID
window from 1980-1989, and evaluates a later window from 2004-2013. Its binary,
32-pixel portrait task is useful for checking temporal reproducibility but does
not offer the visual complexity or operational setting of FMoW.

## CIFAR-10.1 replication

CIFAR-10.1 is a separately collected replacement test set for CIFAR-10. It
provides a small natural-shift check when large WILDS archives are unavailable.
It is not a temporal dataset, so its results are described as cross-collection
reproduction shift.

```bash
uv run fyp-replicate-cifar \
  --data-root data/cifar \
  --output runs/cifar/cifar10-1-screen \
  --train-limit 5000 \
  --eval-limit 1000 \
  --epochs 2
```

The command compares the original CIFAR-10 test set with CIFAR-10.1 v6 using
the same frozen-feature probe, MMD test, domain classifier, and matched baseline
and saliency-gated ViT heads used by the other replication scripts.

The three-seed screen found MMD permutation p = 0.005 for every seed and domain
ROC AUC between 0.589 and 0.606. Mean OOD macro F1 was 73.97% for the uniform
baseline and 74.27% for the saliency head. The mean difference was +0.30 points
with a sample standard deviation of 0.11 points; all three seed differences were
positive. This is a small, repeatable benefit on a natural recollection shift,
not evidence of a large or universal improvement.

## Camelyon17 replication

Camelyon17-WILDS tests hospital acquisition shift in histopathology patches. A
public Hugging Face mirror is used because the official CodaLab certificate has
expired. The mirror groups the source hospitals in one training split, so the
screen creates a class-balanced, patient-disjoint source holdout for ID
evaluation and uses the validation hospital as the OOD sample.

```bash
uv run fyp-replicate-camelyon \
  --output runs/wilds/camelyon17/seed42 \
  --train-limit 5000 \
  --eval-limit 1000 \
  --epochs 2 \
  --seed 42
```

The experiment is a non-standard WILDS screen. It uses an externally pretrained
DINOv2 backbone and three seeds rather than the ten Camelyon17 replicates
requested for a standard leaderboard submission. Both compared heads share
those deviations, so the within-run architecture contrast remains controlled.

Across seeds 42-44, MMD squared ranged from 0.0237 to 0.0566 with permutation
p = 0.005 in every run. Domain ROC AUC ranged from 0.941 to 0.995. Mean OOD
macro F1 was 77.31% for the uniform baseline and 77.24% for the saliency head.
The mean difference was -0.07 points with a sample standard deviation of 0.12.
One run tied and two were slightly negative. Camelyon17 verifies a large
acquisition shift but does not verify an accuracy benefit from the current
saliency head.
