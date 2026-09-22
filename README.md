# Temporal ViT drift experiments

FMoW-WILDS contains satellite images with 62 land-use/building classes and real acquisition years. The official WILDS split separates earlier training images from later evaluation images. The classifier does not receive year or region as an input. [Dataset and split description](https://wilds.stanford.edu/datasets/)

All three experiments use pretrained [DINOv2-small](https://huggingface.co/facebook/dinov2-small), without a CNN. The baseline pools image patches uniformly. XAI learns a saliency gate *inside* the classifier: its patch weights determine which ViT features contribute to the logits. The twin-ViT experiment keeps the older XAI model frozen, adapts a newer XAI ViT on labeled 2013–14 images, and uses class-specific saliency-map differences to flag potential updates.

## Setup and run

```powershell
cd C:\Users\JSW_User\FYP
.\bootstrap.ps1 -Action Setup
.\bootstrap.ps1 -Action Ingest
.\bootstrap.ps1 -Action All -Epochs 3 -BatchSize 8
```

`Setup` installs uv if needed and creates `.venv`. `Ingest` explicitly downloads FMoW-WILDS, which is **about 54 GB compressed** and considerably larger after extraction. A CUDA GPU is strongly recommended. `All` trains the baseline, XAI, and twin-ViT runs after the data is present. Individual actions are `Baseline`, `Xai`, `Siamese`, and `Evaluate`. DINOv2 weights download on first model use.

For a **runtime smoke test without the FMoW download**, run ` .\tools\uv.exe run smoke.py` after setup. It exercises ViT forward/backward passes, metric calculation, saliency routing, and checkpoint/manual snapshot storage using two synthetic images. Its scores are only software checks.

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

The baseline and XAI runs choose `best.pt` by **macro F1 on `id_val`**, which precedes the later adaptation and future test periods. The twin-ViT run adapts on labeled 2013–14 images and chooses its newer model by **macro F1 on 2015**. It then calibrates class-specific routing thresholds on 2015 and evaluates on 2016–17 exactly after model selection. Future test labels never affect checkpoint selection or routing decisions. `last.pt` is saved separately. Model metadata records epoch, selection metric, and score.

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

The twin-ViT report additionally includes update-trigger precision/recall/F1/IoU against the offline diagnostic `old wrong AND new correct`, helpful and harmful route counts, mean saliency Jensen–Shannon divergence, and top-20%-patch saliency overlap IoU/F1. **Saliency overlap measures agreement between two maps, not agreement with ground-truth object regions**: the processed FMoW data used here has no segmentation masks. Each image's classes, map divergence, class threshold, counterfactual evidence, update flag, and routing decision are in `logs/routing_decisions.csv`.

The routing rule is explicit: `D = JS(old_map, new_map)` and `E = P(predicted_class | selected patches) - P(predicted_class | complementary patches)`. Flag an update if the classes differ and `D` exceeds the threshold for the older predicted class. Route to the newer ViT only when it also has `E_new > E_old`. The pairwise training loss keeps maps similar when the older model is correct and encourages a changed map when it is wrong. This is a pseudo-Siamese pair of ViT snapshots; their weights diverge during adaptation. The rule is transparent, but map faithfulness and routing benefit are experimental questions to verify in the reported future-year metrics.

To export overlays, provide explicit run paths:

```powershell
uv run show_saliency.py --checkpoint runs/xai/<run-id>/models/best.pt
uv run show_siamese.py --base-checkpoint runs/xai/<run-id>/models/best.pt --run-dir runs/siamese/<run-id> --index 0
```

The modules stay separate: `ingest.py` downloads, `data.py` loads chronological splits, `model.py` builds ViTs, `train.py` trains baseline/XAI, `siamese.py` defines routing, `train_siamese.py` adapts and tests the twin, `metrics.py` scores, `evaluate.py` evaluates, and `run_store.py` handles run folders and checkpoints.

## Ubuntu server layout

On the Ubuntu server, a checkout at `/home/ubuntu/FYP` can be configured with `bash deploy/remote_setup.sh`. The script installs a project-local Linux `uv`, syncs `uv.lock`, and makes `data`, `runs`, `tools`, and `.venv` symlinks into `/mnt/sdd/FYP` so large files stay on the mounted volume. It refuses to replace existing non-symlink paths. The SSH private key stays on the Windows machine and is never required on the server. The server currently receives this private repository via an SSH-transferred Git bundle, so future updates can use a fresh bundle without adding GitHub credentials there.
