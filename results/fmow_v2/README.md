# fMoW v2 results

`run_outputs/` holds the saved outputs of every v2 run (one JSON per model, condition and seed), copied from
`/mnt/sdd/FYP/runs/v2` on the experiment server. Predictions are stored per test image, so every number in the
paper can be recomputed. `summary_tables.md` lists the headline results.

| Folder | Experiment |
|---|---|
| `main/` | Drift and harm gate, cloud calibration, LoRA rank 8 (2-epoch schedule), frozen baseline |
| `saliency_attention/`, `saliency_soft/` | Attention and saliency heads (3 seeds) |
| `cloud_param/` | Head-width sweep and cloud-aware pooling |
| `shift_test/` | Moved-object test |
| `saliency_learning/` | Saliency in the training signal |
| `saliency_router/` | Auditable router |
| `lora_rank_sweep/` | LoRA ranks 1 and 2 (2-epoch schedule) |
| `token_dropping/`, `strict_saliency/`, `extra_metrics/`, `screens/` | Test-time saliency, calibration and backbone screens |
