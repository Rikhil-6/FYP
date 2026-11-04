# Headline results (macro-F1 %, 2016–17 test)

Heads: mean of 3 seeds. LoRA and hybrid: seed 42 (seeds 43–44 being added).

| Model | Trainable params | Backbone weights changed | Clean | Cumulus beside | Cumulus over | Real clouds |
|---|---|---|---|---|---|---|
| Frozen baseline | 0 | 0 | 49.1 | 46.8 | 42.1 | 44.7 |
| Learned attention | 196,929 | 0 | 60.1 | 58.3 | 49.4 | 55.2 |
| Hard saliency gate | 196,931 | 0 | 59.8 | 57.4 | 48.1 | 55.7 |
| Soft saliency gate | 196,932 | 0 | 60.3 | 58.7 | 49.6 | 55.6 |
| LoRA r1, last 2 blocks | 36,864 | 25.2 M | 57.7 | 56.2 | 50.9 | 56.4 |
| Hybrid (LoRA r1, 2 blocks + soft gate) | 47,124 | 25.2 M | 60.0 | 58.7 | 49.9 | 54.8 |
| LoRA r2, 6 blocks | 221,184 | 75.5 M | 62.7 | 60.7 | 52.0 | 60.1 |
| LoRA r8, 6 blocks | 884,736 | 75.5 M | 62.5 | 60.7 | 52.5 | 59.9 |
