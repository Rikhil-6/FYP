# Figure inputs

| File | Produced by | Used by |
|---|---|---|
| `v2_stats.json` | `../stats.py` from `results/fmow_v2/run_outputs` | most figures |
| `final.json` | `fmow_v2.summarise_equal_training` (server) | equal-training numbers in the paper |
| `analysis_fixed.json` | `fmow_v2.analysis` (server) | real-cloud scores, map stability |
| `examples.npz`, `examples_soft.npz` | `fmow_v2.analysis` / shift-test outputs | `fig_maps` |
| `explain_examples.npz` | `fmow_v2.explain_maps` | `fig_explanations` |
| `explain20.npz` | `fmow_v2.explain_patches` | `fig_explanations` |
