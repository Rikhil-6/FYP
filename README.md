# Explainable ViT routing under temporal drift

This project studies image classification when the deployment data changes over
time. It uses CLEAR-100's year-labelled image stream and three DINOv2-small
variants:

1. a historical ViT that pools patch tokens uniformly;
2. an XAI ViT whose learned patch weights affect the classified representation;
3. a paired historical and updated ViT with a transparent routing rule.

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
- 2010 for adaptation of the updated XAI ViT;
- disjoint class-stratified halves of 2011 for updated checkpoint selection and
  router calibration;
- 2012-2014 for frozen future evaluation.

```bash
tools/uv run fyp-train-router \
  --base-checkpoint runs/xai/<historical-run>/models/best.pt \
  --long-horizon --adaptation-profile realistic --epochs 5
```

The historical model, updated model, and router are frozen before the future
years are scored. No test-year label is used for model selection.

Use `--router-features confidence_only`, `saliency_only`, or `combined` to run
the matched routing-evidence ablation. Each saved policy records the features,
coefficients, scaling values, and thresholds used for its decisions.

## Drift evaluation

The main evaluation reports the natural CLEAR chronology and a controlled suite
covering illumination, white balance, sensor noise, defocus, motion blur, JPEG
compression, resolution loss, viewpoint change, and local occlusion. The
`realistic_mix` condition combines one to three non-occlusion sources and adds
occlusion to a deterministic 10% minority.

```bash
tools/uv run fyp-evaluate-drift \
  --base-checkpoint runs/xai/<historical-run>/models/best.pt \
  --run-dir runs/siamese/<run> --long-horizon
```

The synthetic conditions are controlled sensitivity tests rather than estimates
of deployment frequency. Full definitions and limitations are in
[`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Auditing a route

Classification results include macro F1, macro IoU, accuracy, per-class scores,
and helpful and harmful switch counts. Saliency overlap is reported only as map
agreement; CLEAR-100 does not provide segmentation masks.

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
[`paper/explainable_vit_drift_routing.docx`](paper/explainable_vit_drift_routing.docx).
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
