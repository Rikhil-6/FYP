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
   features. ROC AUC near 0.5 is chance performance; a value towards 1.0 means
   the domains are readily separable.

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

- the clean split-half control estimates within-distribution variation;
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
