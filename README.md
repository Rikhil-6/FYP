# Saliency-guided ViT routing under temporal drift

This project studies image classification when the deployment data changes over
time. It uses CLEAR-100's year-labelled image stream and four DINOv2-small
variants:

1. a historical ViT that pools patch tokens uniformly;
2. a conventionally fine-tuned ViT with the same uniform pooling;
3. a saliency-guided ViT whose learned patch weights affect the classified representation;
4. a paired historical and updated ViT with a transparent routing rule.

The router compares predictions and saliency distributions from the two model
versions. It selects the updated prediction only when the models disagree, the
saliency change clears a calibrated floor, and a logistic policy supports the
switch. Every decision retains the four policy features, threshold, selected
model, predictions, and patch maps.

## Repository layout

```text
src/fyp_drift/     data, models, training, evaluation, and visualisation code
deploy/            Ubuntu setup and logged experiment launchers
scripts/           figure and compact result-generation utilities
tests/             deterministic protocol and checkpoint tests
docs/              architecture, experiment protocol, and server notes
results/           selected metrics and audit examples from the reported run
paper/             editable thesis draft and the principal figures
```

Dataset archives, virtual environments, model checkpoints, caches, and complete
per-image logs are excluded from Git.

## Environment

The project uses [uv](https://docs.astral.sh/uv/) for Python and dependency
management. On Windows:

```powershell
git clone https://github.com/Rikhil-6/FYP.git
cd FYP
git switch Rik/transparent-router
.\bootstrap.ps1 -Action Setup
```

The setup script installs a project-local copy of uv when required and creates
`.venv` from `uv.lock`. On Windows, run commands through
`.\tools\uv.exe run`; on Ubuntu, use `tools/uv run`.

To check the installation without downloading CLEAR-100:

```powershell
.\tools\uv.exe run fyp-smoke
```

The smoke check uses synthetic images. Its scores are software checks and are
not reported as experimental results.

## Data ingestion

CLEAR-100 contains 100 classes arranged by upload year from 2004 to 2014. The
classifier never receives the year as an input; the year defines chronological
training and evaluation windows.

```powershell
.\bootstrap.ps1 -Action Ingest
```

The command downloads the public train and test archives, resumes partial
downloads, validates completed archives, extracts them, and writes
`data/clear100/manifest.csv`. The archives require roughly 4.6 GB.

## Training

A short code-path check is:

```powershell
.\tools\uv.exe run fyp-train --method baseline --epochs 1 --batch-size 2 --limit-train 32 --limit-eval 32
.\tools\uv.exe run fyp-train --method xai --epochs 1 --batch-size 2 --limit-train 32 --limit-eval 32
```

Remove the limits for an experimental run. A CUDA GPU is strongly recommended.
Training uses `tqdm` progress bars and writes configuration, epoch timing,
metrics, predictions, and checkpoints beneath `runs/`.

The long-horizon protocol uses:

- 2004-2008 for historical training;
- 2009 for historical checkpoint selection;
- 2010 for adaptation of the updated saliency-guided ViT;
- disjoint class-stratified halves of 2011 for updated checkpoint selection and
  router calibration;
- 2012-2014 for frozen future evaluation.

```bash
tools/uv run fyp-train --method xai --cv-val-year 2009 --epochs 5

tools/uv run fyp-train-router \
  --base-checkpoint runs/xai/<historical-run>/models/best.pt \
  --long-horizon --adaptation-profile realistic --epochs 5

tools/uv run fyp-train-finetune \
  --base-checkpoint runs/baseline/<historical-run>/models/best.pt \
  --epochs 5 --drift-scenarios clean realistic_mix
```

The historical model, updated model, and router are frozen before the future
years are scored. No test-year label is used for model selection.

`fyp-train-finetune` is the conventional adaptation control. It uses uniform
patch pooling and classification loss only, while retaining the same 2010
adaptation data, 2011 checkpoint-selection split, three trainable backbone
blocks, optimiser settings, and 2012-2014 evaluation horizon as the saliency
experiment.

Use `--router-features confidence_only`, `saliency_only`, or `combined` to run
the matched routing-evidence ablation. Each saved policy records the features,
coefficients, scaling values, and thresholds used for its decisions.

## Drift evaluation

The main evaluation reports the natural CLEAR chronology and a controlled suite
covering illumination, white balance, sensor noise, defocus, motion blur, JPEG
compression, resolution loss, viewpoint change, and local occlusion. The
`realistic_mix` condition applies the top one, two, or three sources selected
on the 2011 calibration year. Sources are ranked by their mean descending MMD
and domain-AUC positions. The frozen top three are motion blur, sensor noise,
and JPEG compression; no 2012-2014 result participates in that choice.

```bash
tools/uv run fyp-evaluate-router-drift \
  --run-dir runs/siamese/<run> \
  --output runs/reevaluation/<run> \
  --scenarios realistic_mix
```

This command loads the saved historical and updated checkpoints and the saved
router policy. It performs evaluation only and records that no training took
place. Use `fyp-evaluate-drift --help` when evaluating one frozen ViT rather
than a saved router.

The synthetic conditions are controlled sensitivity tests rather than estimates
of deployment frequency. Full definitions and limitations are in
[`docs/experiment_protocol.md`](docs/experiment_protocol.md).

Verify that the declared conditions produce measurable distribution shift in
frozen DINOv2 features before interpreting classification changes:

```bash
tools/uv run fyp-verify-drift \
  --data-root data \
  --output runs/drift-verification \
  --limit 500 \
  --permutations 499
```

The command reports RBF maximum mean discrepancy with a permutation p-value and
cross-validated domain-classifier ROC AUC. It includes a clean split-half
control, natural 2009-to-future comparisons, and paired 2014 clean-versus-shift
comparisons for every declared source.

The same checks can be combined with a frozen-feature linear probe on an
official WILDS split. FMoW is the primary replication dataset because its
out-of-distribution split contains satellite images separated by time and
geography. iWildCam and Camelyon17 are available as non-temporal checks of
camera-location and hospital shift respectively.

```bash
tools/uv run fyp-replicate-wilds \
  --dataset fmow \
  --data-root data/wilds \
  --output runs/wilds/fmow \
  --architecture-screen \
  --download
```

The default linear probe is a quick repeatability check. `--architecture-screen`
also trains matched frozen-backbone baseline and saliency-gated ViTs and reports
their ID and OOD scores. This tests the modified classifier on the native WILDS
shift; it does not claim that the full CLEAR-trained router transfers without
dataset-specific updating and calibration.

If the large WILDS downloads are unavailable, the smaller Wild-Time Yearbook
dataset provides an immediate temporal reproducibility check:

```bash
tools/uv run fyp-replicate-yearbook \
  --data-root data/wild-time \
  --output runs/wild-time/yearbook
```

Yearbook is a low-resolution binary task. It verifies the temporal evaluation
and matched architecture comparison, but it is not a substitute for FMoW's
complex satellite imagery.

CIFAR-10.1 provides a second small external screen based on a naturally
recollected test set:

```bash
tools/uv run fyp-replicate-cifar \
  --data-root data/cifar \
  --output runs/cifar/cifar10-1-screen
```

This is a cross-collection reproduction shift rather than a temporal benchmark.
The script reports the same drift tests and matched architecture comparison so
the distinction between measurable shift and predictive harm remains explicit.
Across seeds 42-44, the saliency head improved CIFAR-10.1 OOD macro F1 by a
mean 0.30 points (sample standard deviation 0.11). Yearbook remained negative
by 0.33 points in its single-seed screen, so the external evidence is mixed.

Camelyon17-WILDS adds a real acquisition shift across hospitals. The public
mirror can be streamed, so the screen does not require the expired CodaLab
certificate or a complete 10.7 GB download:

```bash
tools/uv run fyp-replicate-camelyon \
  --output runs/wilds/camelyon17/seed42 \
  --train-limit 5000 \
  --eval-limit 1000 \
  --epochs 2 \
  --seed 42
```

The source-hospital training and ID samples are class balanced and patient
disjoint. The OOD sample comes from the validation hospital. This is an
exploratory matched-head screen: DINOv2 pretraining makes it a non-standard
WILDS run, and the WILDS guidance recommends ten Camelyon17 replicates for a
leaderboard result. Across seeds 42-44, the hospital shift was detected in every
run, but the saliency head changed OOD macro F1 by only -0.07 points on average
(sample standard deviation 0.12). The third dataset strengthens the drift
detection evidence while rejecting a general accuracy-improvement claim.

Summarise completed external screens without pooling unlike datasets:

```bash
tools/uv run fyp-summarise-replications \
  results/drift_verification/yearbook_replication.json \
  results/drift_verification/cifar10_1_seed*.json \
  results/drift_verification/camelyon17_seed*.json \
  --output results/drift_verification
```

ACDC provides a further, more realistic adverse-driving check based on paired
normal, fog, night, rain and snow scenes. It is a semantic segmentation dataset,
so it is not treated as a drop-in classification replacement for CLEAR. After
registering with the official site and extracting the two archives, validate
the pairs with:

```bash
tools/uv run fyp-ingest-acdc --data-root data/ACDC
```

The paired-condition audit reports whether adaptation reduces MMD, domain
separability, matched-scene feature distance and prediction divergence while
checking feature variance and effective rank for collapse. It also records
layer-wise checkpoint movement. The JSON separates baseline drift detection,
representation mitigation, residual drift and task mitigation, so a partial
reduction is not reported as elimination. See
[`docs/acdc_adaptation_protocol.md`](docs/acdc_adaptation_protocol.md) for the
required evidence and the command format.

## Auditing a route

Classification results include macro F1, macro IoU, accuracy, per-class scores,
and helpful and harmful switch counts. Saliency overlap is reported only as map
agreement; CLEAR-100 does not provide segmentation masks.

The class-level export keeps low-support classes visible while excluding them
from aggregate gain claims. It also ranks eligible classes by historical F1 and
splits them into four equal groups, which shows whether changes concentrate in
the weakest or strongest historical classes.

```bash
tools/uv run fyp-export-classwise \
  --decisions runs/siamese/<run>/logs/drift_routing_decisions.csv \
  --class-names data/clear100/classes.json \
  --output-dir results/classwise/<run> \
  --minimum-support 20
```

For cases where the two models differ in correctness, the audit removes the top
and bottom 20% of patches under each map. A positive deletion margin means that
removing highly weighted patches reduces true-class probability more than
removing weakly weighted patches.

```bash
tools/uv run fyp-audit-router \
  --base-checkpoint runs/xai/<historical-run>/models/best.pt \
  --run-dir runs/siamese/<run> \
  --output runs/audits/drift
```

This is a prediction-sensitivity check. It does not prove causal attribution or
object localisation.

## Reported results

The compact result set is under [`results/`](results/). It includes the matched
2014 drift table, ablation table, source-specific audit summaries, and selected
saliency panels. Large checkpoints and complete decision logs remain outside
Git.

The editable paper is
[`paper/saliency_guided_vit_drift_routing.docx`](paper/saliency_guided_vit_drift_routing.docx).
The main architecture and result figures are also stored separately under
[`paper/figures/`](paper/figures/).

## Server execution

`deploy/remote_setup.sh` configures an Ubuntu checkout and keeps large data,
caches, runs, and the virtual environment on the mounted data volume. The
server launcher checks the locked PyTorch build, CUDA visibility, and an actual
GPU tensor operation before training.

```bash
bash deploy/server.sh status
bash deploy/server.sh setup
bash deploy/server.sh ingest
bash deploy/launch.sh --detach train --temporal-repeats 1 --epochs 5 --batch-size 8
```

See [`docs/server_setup.md`](docs/server_setup.md) for the storage layout,
logging paths, and tmux commands. The source package and command map are
described in [`docs/architecture.md`](docs/architecture.md).
