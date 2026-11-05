# fMoW v2 results

Everything here is generated from the raw per-seed runs; nothing is typed in by hand.

| File or folder | What it is |
|---|---|
| `run_outputs/` | Saved output of every run (one JSON per model, condition and seed, with per-test-image predictions), copied from the experiment server |
| `test_labels.npz` | Class, cloud cover and "site seen earlier" flag of the 7,077 test images (from `fmow_v2.export_test_labels`), so tables can be rebuilt without the images |
| `results_table.json`, `summary_tables.md` | Headline results (three-seed means, graded cloud exposure, real-cloud and class-matched scores, hashes of every input), written by `python -m fyp_drift.evaluation.fmow_v2.aggregate` |
| `site_overlap.json` | Cross-period site audit (`fmow_v2.site_overlap`) |

| `run_outputs/` folder | Experiment |
|---|---|
| `main/` | Drift and harm gate, cloud calibration, LoRA rank 8 (2-epoch schedule), frozen baseline |
| `saliency_attention/`, `saliency_soft/` | Attention and saliency heads (3 seeds) |
| `heldout_*` | The same heads on held-out cumulus clouds beside / over the object |
| `lora_fair/` | LoRA r1 and r2 on the last two blocks, r2 and r8 on six blocks (3 seeds), same recipe as the hybrid |
| `hybrid/` | Hybrid (LoRA + width-8 scorer): soft-gated rank 1 and 2, and the saliency-free control (`clean_attention_*`) |
| `control_hist/` | Heads trained on 2010-13 instead of 2014-15, and graded cloud exposure (`mix.json`) |
| `cloud_param/` | Head-width sweep and cloud-aware pooling |
| `shift_test/` | Moved-object test |
| `saliency_learning/`, `saliency_router/` | Saliency in the training signal; auditable router |
| `lora_rank_sweep/` | LoRA ranks 1 and 2 (2-epoch schedule) |
| `token_dropping/`, `strict_saliency/`, `extra_metrics/`, `screens/`, `analysis/`, `heads_lr*/` | Test-time saliency, calibration, learning-rate and backbone screens |
