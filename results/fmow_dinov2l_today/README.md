# fMoW DINOv2-L study: raw results

Outputs of the cloud-cover and saliency studies, copied from the server run
directory `runs/today`. They were produced by the code at commit `fca59db` on
branch `fyp/fmow-dinov2l-today`, on 2026-11-03.

Every number in the top-level README comes from a file here. All studies use
three seeds (42, 43, 44) and score on the 2016-17 test split.

## Contents

- `main/`: drift and harm gate, cloud calibration, LoRA arms and the
  unseen-cloud stage (`fmow_dinov2l_study`).
- `saliency_attention/`: five attention heads with shuffled twins, plus
  faithfulness and grounding tests (`saliency_attention_study`).
- `saliency_soft/`: softer heads that let context through
  (`soft_saliency_heads`).
- `extra_metrics/`: worst-region error, calibration, box IoU, and deletion and
  insertion curves (`extra_metrics`).
- `lora_rank_sweep/`: LoRA at ranks 1, 2 and 8, matched to the heads
  (`lora_rank_sweep`).
- `saliency_router/`: router summaries, policies and per-image route logs
  (`saliency_router_study`).
- `token_dropping/`: test-time token dropping under cloud cover
  (`fmow_token_dropping`).
- `logs/`: console logs of the runs above.
- `cloud_previews/`: examples of the synthetic cloud layer.
- `run_scripts/`: the exact commands that launched the main runs.

## Reading the files

- Each folder has a `summary.json` with the per-condition table, where one
  exists. Per-seed files are named `<condition>_<arm>_s<seed>.json`.
- `main/gate.json` holds the drift tests and the harm gate. `main/calib.json`
  holds the cloud strength chosen on historical validation data.
- Route logs in `saliency_router/` are gzipped CSV files with one row per test
  image: the policy signals, the switch decision and the outcome.

## Not included

- Frozen feature caches, the linear head checkpoint and per-image attention
  maps (`*.npz`, `*.pt`), which are large and can be regenerated.
- The earlier frozen-token experiments (saliency adapters, guidance sweep and
  strict pooling) and the cross-dataset harm screens.
