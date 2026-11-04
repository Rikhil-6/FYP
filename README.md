# Saliency-guided attention on a frozen Vision Transformer under temporal drift

Final-year project. A satellite-image classifier is trained on 2010–13 images and must be updated for 2016–17,
under natural drift and cloud cover. The project asks whether the old model's own patch-level saliency can make
that update better, cheaper or easier to audit, and compares it with learned attention and LoRA.

![System overview](figures/fmow_v2/figures/fig_overview.png)

## Headline results

Macro-F1 (%) on 2016–17 fMoW test images, models trained on clean 2014–15 images
(heads: mean of 3 seeds; LoRA and hybrid: seed 42).

| Model | Trainable params | Backbone weights changed | Clean | Cloud beside object | Cloud over object | Real clouds |
|---|---|---|---|---|---|---|
| Frozen baseline | 0 | 0 | 49.1 | 46.8 | 42.1 | 44.7 |
| Learned attention | 197k | 0 | 60.1 | 58.3 | 49.4 | 55.2 |
| Hard saliency gate | 197k | 0 | 59.8 | 57.4 | 48.1 | 55.7 |
| Soft saliency gate | 197k | 0 | 60.3 | 58.7 | 49.6 | 55.6 |
| Hybrid: LoRA r1 (2 blocks) + soft gate | 47k | 25 M | 60.0 | 58.7 | 49.9 | 54.8 |
| LoRA r2 (6 blocks) | 221k | 75 M | **62.7** | **60.7** | **52.0** | **60.1** |

- **Accuracy:** LoRA is the most accurate update; attention heads come within about 2.5 points without touching
  the backbone and train in about a minute.
- **Saliency** adds neither accuracy nor cloud robustness (shuffled-map controls match every saliency head).
- **What saliency does give** is explanation: both the old and the updated heads decompose *exactly* into
  per-patch contributions, and saliency-gated heads put most of their support on the object.

![Accuracy of the main models](figures/fmow_v2/figures/fig_accuracy.png)
![Explanations before and after an update](figures/fmow_v2/figures/fig_explain_update.png)

## Repository layout

| Path | Contents |
|---|---|
| `src/fyp_drift/` | Python package: data, models, training and all experiments (`evaluation/`, v2 additions in `evaluation/fmow_v2/`) |
| `models/fmow_v2/` | Trained models (safetensors) with a one-line inference CLI |
| `results/fmow_v2/` | Saved output of every v2 run, with per-image predictions |
| `figures/fmow_v2/` | Every figure, and the code + data that draw it |
| `scripts/` | Run scripts (`fmow_v2/`) and early pilots (`pilots/`) |
| `docs/` | Guides (below) and earlier protocol notes |
| `tests/` | Unit tests (`pytest`) |
| `deploy/` | Server setup and batch scripts |

The earlier CLEAR-100 version of the project (transparent router) is preserved on the `initial` branch.

## Guides

| Guide | For |
|---|---|
| [Getting started](docs/getting_started.md) | Install, download the data, run, rebuild figures |
| [Data](docs/data.md) | fMoW source, subset, splits, preparation, clouds |
| [Code map](docs/code_map.md) | What each part of the code does, with an architecture diagram |
| [Experiment run order](docs/fmow_v2.md) | Every v2 step, its command and output, protocol notes |
| [Trained models](docs/models.md) | Saved models, inference, re-exporting |

## Quick start

```bash
pip install -e . && pip install seaborn scipy safetensors
python -m fyp_drift.evaluation.fmow_v2.models predict models/fmow_v2/soft_gate my_site.jpg --map attention.png
```
