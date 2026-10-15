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
   features. The reported AUC is symmetrised around 0.5 so that reversing the
   arbitrary domain labels does not change the result. A value near 0.5 is
   chance performance; a value towards 1.0 means the domains are readily
   separable.

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
