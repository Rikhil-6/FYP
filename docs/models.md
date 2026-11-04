# Trained models

Saved in `models/fmow_v2/` as **safetensors** (weights only: safe to load, no pickle code execution, memory-mapped
and fast) plus a `config.json` with the class names, architecture and validation score. The frozen DINOv2-L
backbone is not stored; it downloads from the Hugging Face hub on first use (`facebook/dinov2-large`).

| Folder | Model | Trainable params | Clean test macro-F1 (seed 42) |
|---|---|---|---|
| `historical/` | Frozen 2010–13 linear head | – | 49.1 |
| `attention/` | Learned attention head | 196,929 | 59.9 |
| `hard_gate/` | Hard saliency gate | 196,931 | 59.4 |
| `soft_gate/` | Soft saliency gate | 196,932 | 60.4 |
| `lora_r2/` | LoRA rank 2, 6 blocks (adapters only) | 221,184 | ≈62.7 |
| `hybrid/` | LoRA r1, last 2 blocks + soft gate | 47,124 | ≈60.0 |

## Inference

```bash
python -m fyp_drift.evaluation.fmow_v2.predict models/fmow_v2/soft_gate site.jpg
python -m fyp_drift.evaluation.fmow_v2.predict models/fmow_v2/soft_gate site.jpg --box 0.52,0.48,0.30,0.22 --map attention.png
```

`--box` (object centre x, y, width, height as fractions of the image) crops the image the way the models were
trained; without it the whole image is used. `--map` saves the head's 10 x 10 attention over the image.

From Python:

```python
from fyp_drift.evaluation.fmow_v2.models import Predictor

predictor = Predictor.load("models/fmow_v2/soft_gate")   # loads backbone + head once
result = predictor.predict("site.jpg")                   # {"top": [(class, prob), ...], "weights": 10x10 array}
```

Loading takes a few seconds (mostly the backbone); each image then takes well under a second on a GPU.

## Re-exporting

```bash
python -m fyp_drift.evaluation.fmow_v2.export_heads --out models/fmow_v2                     # heads, ~1 min each
python -m fyp_drift.evaluation.fmow_v2.equal_lora --rank 2 --seed 42 --export models/fmow_v2/lora_r2
python -m fyp_drift.evaluation.fmow_v2.hybrid --seed 42 --export models/fmow_v2/hybrid
```
