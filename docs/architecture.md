# Code structure

The project is an installable Python package. Run every Python stage through
`uv run` so local and server executions use the locked environment.

## Package map

```text
src/fyp_drift/
├── data/             CLEAR-100 ingestion, manifests, temporal splits, drift transforms
├── models/           DINOv2 classifier, Siamese comparison, transparent routing policy
├── training/         Historical/XAI training and temporal router adaptation
├── evaluation/       Metrics, frozen evaluations, saliency analysis, counterexample audits
├── artifacts/        Checkpoints, run folders, manual snapshots, report exports
├── visualization/    Drift previews and saliency or router panels
├── workflows/        Multi-stage experiment orchestration
└── system/           Hardware inspection and the synthetic runtime check
```

The folders describe responsibility rather than experiment order. Training code
loads data through `data/`, builds networks from `models/`, writes through
`artifacts/`, and calls metric functions from `evaluation/`. Dataset downloads do
not occur inside a training command.

## Main commands

| Stage | Command | Main output |
| --- | --- | --- |
| Environment | `uv sync --locked` | `.venv/` |
| Data ingestion | `uv run fyp-ingest` | `data/clear100/manifest.csv` |
| Baseline or XAI training | `uv run fyp-train --method baseline` | `runs/baseline/<run-id>/` |
| Conventional temporal fine-tuning | `uv run fyp-train-finetune --base-checkpoint <path>` | `runs/finetune/<run-id>/` |
| Temporal adaptation and routing | `uv run fyp-train-router --base-checkpoint <path>` | `runs/siamese/<run-id>/` |
| Frozen classifier evaluation | `uv run fyp-evaluate ...` | JSON metrics and prediction CSV |
| Controlled drift evaluation | `uv run fyp-evaluate-drift ...` | Per-source temporal metrics |
| Class-level router analysis | `uv run fyp-export-classwise ...` | Per-class and historical-difficulty CSV tables |
| Counterexample audit | `uv run fyp-audit-router ...` | Decision CSV and bounded panels |
| Runtime check | `uv run fyp-smoke` | Temporary synthetic run |
| Hardware check | `uv run fyp-hardware` | Torch, CUDA, GPU, CPU, and storage status |

Run `uv run <command> --help` for the full arguments. `pyproject.toml` is the
single command registry used on Windows and Ubuntu.

## Experiment boundary

The chronological pipeline has five separate decisions:

1. Fit the historical classifier on earlier years.
2. Select its checkpoint on a later validation year.
3. Adapt either a uniform-pooling ViT with classification loss only or a copied
   XAI ViT on the declared adaptation window.
4. Select the updated checkpoint and fit the router on disjoint calibration data.
5. Freeze both models and the router before evaluating future years.

`training/classifier.py` owns steps 1 and 2. `training/fine_tune.py` provides the
conventional adaptation control. `training/temporal_router.py` owns the
saliency-guided adaptation and routing path. Metrics and figures consume saved
artifacts; they do not retrain a model.

## Stored artifacts

Each run has the same small set of directories:

```text
<run-id>/
├── config.json
├── logs/
├── metrics/
├── models/
└── manual/
```

`models/best.pt` records the validation-selected weights. `models/last.pt`
records the final epoch. Temporal router runs also update `models/resume.pt`
atomically after every completed epoch. Routing CSV files retain the four policy
features, the route probability, the chosen tower, and both model predictions.

## Deployment scripts

`bootstrap.ps1` prepares and runs the project on Windows. `deploy/server.sh`
provides the matching Ubuntu commands. `deploy/launch.sh` adds logged `tmux`
execution for long jobs. The scripts call package entry points and contain no
duplicate model logic.
