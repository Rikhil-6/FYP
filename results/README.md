# Reported experiment results

This directory contains a compact, reviewable subset of the reported runs. The
complete model checkpoints, event logs, and per-image decision tables remain in
server storage.

## Five-epoch ablation

[`ablation.csv`](ablation.csv) compares three runs with the same seed, five-epoch
budget, historical checkpoint, adaptation years, and frozen 2012-2014 horizon.

| Configuration | 2014 combined updated F1 | 2014 combined routed F1 | Helpful | Harmful |
| --- | ---: | ---: | ---: | ---: |
| Matched standard | 91.68 | 91.72 | 144 | 57 |
| Enhanced with saliency losses | **91.99** | **91.85** | **188** | 90 |
| Enhanced without saliency losses | 91.86 | 91.62 | 125 | **43** |

The auxiliary saliency objectives add 0.13 points to updated-model macro F1 and
0.23 points to routed macro F1 relative to the matched no-auxiliary control.
They also increase route coverage, producing both more helpful and more harmful
switches.

## Source-separated drift

[`drift/metrics_2014.csv`](drift/metrics_2014.csv) reports all eleven 2014
conditions. The saliency-supervised router gains 3.45 points over the frozen
historical ViT under motion blur and 2.03 points under the realistic mixture.
Relative to the matched router without auxiliary saliency losses, the additional
routed gains are 0.90 and 0.84 points respectively.

The clean routed score is 0.18 points lower than the no-auxiliary control. Mild
conditions also show small or mixed effects, so the saliency contribution is
reported as source dependent.

## External screens

`drift_verification/` contains the Yearbook, CIFAR-10.1 and Camelyon17 run
records plus a combined CSV and JSON summary. The matched saliency head changes
OOD macro F1 by -0.33 points on Yearbook, +0.30 +/- 0.11 points across three
CIFAR-10.1 seeds, and -0.07 +/- 0.12 points across three Camelyon17 seeds.
Camelyon17 shows strong hospital separation in frozen DINOv2 features, with
domain AUC from 0.941 to 0.995, but no corresponding saliency-head benefit.

## Route audits

The `audits/` folders retain aggregate JSON, selected-case CSV files, and two
helpful, two harmful, and two missed examples for motion blur, viewpoint shift,
and the realistic mixture. The deletion audit compares the effect of removing
the most and least weighted patches.

The maps are evidence-allocation diagnostics. They are not segmentation masks,
and a positive deletion margin does not establish that highlighted pixels caused
the prediction.
