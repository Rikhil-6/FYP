# Temporal ViT drift experiments

CLEAR-100 contains real-world images in 100 classes, organized into upload-year buckets from 2004 through 2014. Its publicly downloadable train and test archives are linked from the [benchmark's official download page](https://clear-benchmark.github.io/). The classifier never receives the year as an input; years define chronological training, adaptation, and evaluation periods.

All three experiments use pretrained [DINOv2-small](https://huggingface.co/facebook/dinov2-small), without a CNN. The baseline pools image patches uniformly. XAI learns a saliency gate *inside* the classifier: its patch weights determine which ViT features contribute to the logits. The twin-ViT experiment keeps the older XAI model frozen, adapts a newer XAI ViT on 2012 images, and uses class-specific saliency-map differences to flag potential updates.

## Setup and run

```powershell
cd C:\Users\JSW_User\FYP
.\bootstrap.ps1 -Action Setup
.\bootstrap.ps1 -Action Ingest
.\bootstrap.ps1 -Action All -Epochs 3 -BatchSize 8
```

`Setup` installs uv if needed and creates `.venv`. `Ingest` downloads the two CLEAR-100 image archives, about **4.6 GB combined**, and prepares a local manifest. A CUDA GPU is strongly recommended. `All` trains the baseline, XAI, and twin-ViT runs after the data is present. Individual actions are `Baseline`, `Xai`, `Siamese`, and `Evaluate`. DINOv2 weights download on first model use.

For a **runtime smoke test without CLEAR-100**, run `.\tools\uv.exe run smoke.py` after setup. It exercises ViT forward/backward passes, metric calculation, saliency routing, and checkpoint/manual snapshot storage using two synthetic images. Its scores are only software checks.

After ingestion, a small code-path run is:

```powershell
uv run train.py --method baseline --epochs 1 --batch-size 2 --limit-train 32 --limit-eval 32
uv run train.py --method xai --epochs 1 --batch-size 2 --limit-train 32 --limit-eval 32
```

These limits are only for debugging. Do not interpret their scores as research results.

## Run folders and checkpoints

Each invocation creates an isolated folder such as `runs/xai/20260922T120000Z/` or `runs/siamese/20260922T120000Z/`:

```text
config.json
logs/events.jsonl
logs/predictions.csv               # baseline/XAI
logs/routing_decisions.csv         # twin-ViT
metrics/final.json                 # baseline/XAI
metrics/future_test.json           # twin-ViT
metrics/thresholds.json            # twin-ViT
models/best.pt
models/best.json
models/last.pt
models/last.json
manual/
```

The baseline and XAI runs choose `best.pt` by **macro F1 on `id_val`** (2011), which precedes adaptation and future testing. The twin-ViT run adapts on labeled 2012 images, chooses its newer model and calibrates class-specific routing thresholds on 2013, and evaluates on held-out 2014 images. Future test labels never affect checkpoint selection or routing decisions. `last.pt` is saved separately. Model metadata records epoch, selection metric, and score.

To manually preserve a selected checkpoint and its metadata in the run's `manual/` folder:

```powershell
uv run save_model.py --run-dir runs/xai/<run-id> --name thesis_candidate --which best
```

Use the actual run ID printed by training. Manual snapshots do not overwrite an existing name. A standalone evaluation can be written to a separate destination:

```powershell
uv run evaluate.py --method xai --checkpoint runs/xai/<run-id>/models/best.pt --output runs/xai/<run-id>/metrics/recheck
```

## Metrics and interpretation

`metrics/final.json` and `metrics/future_test.json` report accuracy, macro precision/recall/F1, weighted F1, macro IoU, per-class precision/recall/F1/IoU/support, and a confusion matrix. Scores are broken out by real year where applicable. **Classification IoU** uses `TP / (TP + FP + FN)` for each class; it is not image-segmentation IoU.

The twin-ViT report additionally includes update-trigger precision/recall/F1/IoU against the offline diagnostic `old wrong AND new correct`, helpful and harmful route counts, mean saliency Jensen–Shannon divergence, and top-20%-patch saliency overlap IoU/F1. **Saliency overlap measures agreement between two maps, not agreement with ground-truth object regions**: CLEAR-100 provides class labels rather than segmentation masks. Each image's classes, map divergence, class threshold, counterfactual evidence, update flag, and routing decision are in `logs/routing_decisions.csv`.

The routing rule is explicit: `D = JS(old_map, new_map)` and `E = P(predicted_class | selected patches) - P(predicted_class | complementary patches)`. Flag an update if the classes differ and `D` exceeds the threshold for the older predicted class. Route to the newer ViT only when it also has `E_new > E_old`. The pairwise training loss keeps maps similar when the older model is correct and encourages a changed map when it is wrong. This is a pseudo-Siamese pair of ViT snapshots; their weights diverge during adaptation. The rule is transparent, but map faithfulness and routing benefit are experimental questions to verify in the reported future-year metrics.

## Controlled deployment drift and timing

Epoch records now include wall time, CUDA event time, training images/second, and validation inference latency. Every evaluated split also stores wall time, CUDA time, images/second, and milliseconds/image under `_timing`. Wall time is the operational measurement because it includes image loading and preprocessing; CUDA time isolates work queued on the GPU.

Controlled drift is opt in and disabled during ordinary training. `evaluate_drift.py` applies deterministic, evaluation-only covariate shifts to the later timeline: severity 1 in 2012, severity 2 in 2013, and severity 3 in 2014. Reports include a clean reference plus reduced saturation, low Gaussian sensor noise, a small local-mean occlusion, and their combination. The configured levels are deliberately mild (saturation factors 0.90/0.80/0.70, noise standard deviations 0.01/0.02/0.035 on a `[0,1]` image, and occluded areas 2%/5%/8%). Training images and labels are never modified.

These choices follow the common-corruption evaluation protocol introduced by [ImageNet-C](https://arxiv.org/abs/1903.12261), including noise and saturation at ordered severities. [AugMix](https://arxiv.org/abs/1912.02781) motivates evaluation against unforeseen corruptions, while the [IRUO occlusion study](https://doi.org/10.1016/j.patcog.2025.112215) reports that controlled synthetic occlusion can serve as a useful proxy for real occlusion. This suite measures sensitivity; it does not claim that synthetic corruptions reproduce every deployment shift.

Run it on the latest baseline and XAI checkpoints:

```bash
bash deploy/server.sh drift --batch-size 16
```

Results are saved beside each run at `metrics/drift/drift_evaluation.json`. Use `--limit 100` for a quick execution check. To include drift evaluation automatically after a future complete experiment, pass `--evaluate-drift` to `deploy/server.sh experiment`; omitting it performs no corruption evaluation. With the flag enabled, the Siamese router is also evaluated on the corrupted 2014 stream and writes `metrics/future_drift.json`.

To export overlays, provide explicit run paths:

```powershell
uv run show_saliency.py --checkpoint runs/xai/<run-id>/models/best.pt
uv run show_siamese.py --base-checkpoint runs/xai/<run-id>/models/best.pt --run-dir runs/siamese/<run-id> --index 0
```

The modules stay separate: `ingest.py` downloads, `data.py` loads chronological splits, `model.py` builds ViTs, `train.py` trains baseline/XAI, `siamese.py` defines routing, `train_siamese.py` adapts and tests the twin, `metrics.py` scores, `evaluate.py` evaluates, and `run_store.py` handles run folders and checkpoints.

## Ubuntu server layout

On the Ubuntu server, a checkout at `/home/ubuntu/FYP` can be configured with `bash deploy/remote_setup.sh`. The script installs a project-local Linux `uv`, syncs `uv.lock`, and makes `data`, `runs`, `tools`, and `.venv` symlinks into `/mnt/sdd/FYP` so large files stay on the mounted volume. It refuses to replace existing non-symlink paths. The SSH private key stays on the Windows machine and is never required on the server. The server currently receives this private repository via an SSH-transferred Git bundle, so future updates can use a fresh bundle without adding GitHub credentials there.

Use `bash deploy/server.sh status` to check the Git revision, PyTorch GPU visibility, and resolved symlinks. `bash deploy/server.sh smoke` runs the tiny synthetic check. `bash deploy/server.sh ingest` downloads and indexes CLEAR-100. `baseline`, `xai`, `siamese`, `evaluate`, and `all` dispatch their corresponding experiments. Full training requires a GPU by default; set `FYP_ALLOW_CPU_TRAIN=1` explicitly to proceed on a CPU-only machine. The runner keeps uv and Hugging Face caches on `/mnt/sdd/FYP/cache`.

The server experiment launcher checks that the installed CUDA enabled PyTorch matches `uv.lock` and performs a real GPU tensor operation before training. It also runs a synthetic DINOv2 forward/backward smoke test, then trains the requested chronological folds and final baseline/XAI/twin models. Each batch has a `tqdm` progress bar; batch and epoch records go into each model's `logs/events.jsonl`. The combined terminal output is saved in `runs/experiments/<timestamp>/console.log`, and its `summary.json` lists the model run paths and fold scores. Keep an SSH session open or run it under `tmux` for long jobs:

```bash
cd /home/ubuntu/FYP
bash deploy/server.sh ingest
tmux new -s fyp
bash deploy/server.sh experiment --temporal-repeats 3 --epochs 3 --batch-size 8
```

Temporal backtesting keeps the fit period fixed: every repeat trains on 2004–2010, selects its checkpoint on 2011, and scores the unchanged checkpoint separately on 2012 and 2013. No later year is folded back into baseline training. `--temporal-repeats 3` repeats that protocol with three seeds; `--cv-folds` remains a compatibility alias. The final untouched test is 2014. For a short code-path check, add `--cv-only --temporal-repeats 1 --epochs 1 --limit-train 32 --limit-eval 32`; those sample scores are only for debugging. Detach from `tmux` with Ctrl+B then D, and reconnect with `tmux attach -t fyp`.
Before downloading CLEAR-100, `bash deploy/server.sh experiment --smoke-only` verifies the installed PyTorch CUDA build and runs the ViT smoke path through the same logged launcher.

To start ingestion followed by training in one detached server job, while keeping each stage separate and logged, run:

```bash
cd /home/ubuntu/FYP
bash deploy/launch.sh --detach all --temporal-repeats 3 --epochs 3 --batch-size 8
```

The command prints a `tmux` session name. `bash deploy/launch.sh ingest` runs ingestion alone, and `bash deploy/launch.sh train --temporal-repeats 3 --epochs 3 --batch-size 8` runs training on existing data. Stage logs go to `runs/pipelines/<timestamp>/ingest.log` and `train.log`; model metrics and checkpoints retain their own run folders.
Ingestion resumes a partial archive download, skips valid archives and already extracted images, and skips all work after `manifest.csv` and extraction markers are present. The manifest records each file's class, source archive, bucket, year, and split. The chronological protocol is:

| Split | CLEAR-100 source | Year |
| --- | --- | --- |
| Train | Train buckets 0–6 | 2004–2010 |
| ID validation | Train bucket 7 | 2011 |
| ID test | Test bucket 7 | 2011 |
| Adaptation and calibration | Train buckets 8–9 | 2012–2013 |
| Future test | Test bucket 10 | 2014 |

The setup and runner derive their project path from their own file location. A separate Git worktree can therefore use `FYP_STORAGE_DIR=/mnt/sdd/FYP-fork FYP_DATA_DIR=/mnt/sdd/FYP/data FYP_CACHE_DIR=/mnt/sdd/FYP/cache FYP_TOOLS_DIR=/mnt/sdd/FYP/tools bash deploy/server.sh setup` to share the downloaded dataset, uv package cache, and uv executable while keeping its own runs and virtual environment on the mounted volume.

## Transparent router fork

Branch `Rik/transparent-router` keeps combined corruption entirely held out until the 2014 test. The router is calibrated only on 2013 labels using four logged features: saliency Jensen-Shannon divergence, confidence gain, entropy reduction, and counterfactual evidence gain. A standardized logistic rule and an explicit saliency floor choose whether to use the updated ViT. `metrics/router_policy.json` records every coefficient, scale, and threshold.

`export_router_report.py` writes three bounded four-panel examples for every condition (clean, saturation, Gaussian noise, occlusion, and combined) plus `samples.json`. Store these under `reports/transparent_router/<run-id>/` so the visual evidence can be committed without adding the full dataset or thousands of generated images.

### Long-horizon protocol

Pass `--long-horizon` to freeze the chronology as follows: historical fit 2004-2008, checkpoint selection 2009, drift-aware adaptation 2010, router calibration 2011, and untouched test-source evaluation in 2012-2014. Adaptation and calibration use a deterministic 50% clean / 50% single-corruption mixture; combined corruption is excluded until the future test. The model is not updated between test years. `evaluate_drift.py --long-horizon` and `export_router_report.py --long-horizon` use the same three-year horizon.

Add `--adaptation-profile enhanced` to test a predeclared stronger correction policy. It exposes the newer ViT to a deterministic 25% clean / 75% individual-corruption curriculum spanning severities 1-3, trains its final three transformer blocks, uses separate conservative backbone and head learning rates, and increases the saliency faithfulness loss. Combined corruption remains unseen until the frozen future test. Router calibration jointly selects one of four saliency-floor quantiles and its route-probability threshold on 2011 only. Compare this run against `standard` with the same historical checkpoint, seed, epoch count, calibration year, and 2012-2014 test horizon.

Every twin-ViT epoch now atomically updates `models/resume.pt` and `models/resume.json` with model, optimizer, epoch, and best-score state. Resume an interrupted run without overwriting its earlier events:

```bash
uv run train_siamese.py --base-checkpoint runs/xai/<run>/models/best.pt \
  --resume-run-dir runs/siamese/<interrupted-run> --long-horizon --epochs 5
```

For a matched capacity/data control, add `--disable-saliency-loss`. Use `--drift-scenarios clean combined` for a focused ablation while retaining all five conditions in the main run. New drift evaluations save `logs/drift_routing_decisions.csv`. Summarize helpful/harmful route saliency and generate a 2014 combined-drift outcome figure with:

```bash
uv run --with matplotlib analyze_saliency.py \
  --run-dir runs/siamese/<run> --output reports/saliency/<run>
```

Audit the complete 2014 compound-drift stream for corrections, regressions, missed
updates, and saliency deletion faithfulness with:

```bash
uv run --no-sync audit_counterexamples.py \
  --base-checkpoint runs/xai/<historical-run>/models/best.pt \
  --run-dir runs/siamese/<enhanced-run> \
  --data-root data \
  --output reports/transparent_router/<audit-name>
```

The audit keeps each manifest index, writes every decision to CSV, applies an
equal-area top-versus-bottom 20% patch deletion check to every sample where the
old and new models differ in correctness, and exports bounded panels for helpful,
harmful, and missed routes. A positive deletion margin means that removing the
model's most salient patches reduced true-class probability more than removing
its least salient patches. This is a prediction-sensitivity check rather than an
object-segmentation score or causal proof.
