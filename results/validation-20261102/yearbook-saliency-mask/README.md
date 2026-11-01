# Chronological Yearbook saliency-mask pilot, seed 42

Run on `ubuntu@rik.nus.cloud` in `/home/ubuntu/FYP-validation-20261031` on a Tesla T4. The source file at run time was `src/fyp_drift/training/yearbook_saliency_mask_pilot.py`, SHA-256 `06782a1c5e2257025af031cf6f19732293b9e76d88c6991e49907c1def6b47bf`. The full server run is `runs/yearbook-saliency-mask-pilot-20261102`; the four 85 MB checkpoints remain there. `summary.json` SHA-256: `acf1c479aaa2911034cd785a76640ad0879a29c00f60c110023ec0d9729accb7`.

## Split and controls

- Historical head: train on 1930–1989 native training images, choose epoch on 1980–1989 native test images.
- Adaptation: train on 1990–1999 native training images; choose each arm's epoch on 2000–2003 native test images.
- Final future evaluation: 2004–2013 native test images, 335 total. No future labels tune training or selection.
- Adaptation arms: ordinary rank-4 LoRA, random equal-area masking plus masked-view CE, saliency top-40%-patch masking plus masked-view CE. The latter two have the same trainable parameter count (19,202), epochs and sample IDs, and the same second-view loss weight (0.5). The plain LoRA arm has less forward-pass compute.

| Model | Future correct / 335 | Future accuracy | Future macro F1 |
| --- | ---: | ---: | ---: |
| Historical, no adaptation | 312 | 93.13% | 93.13% |
| Ordinary LoRA | 322 | 96.12% | 96.12% |
| Random mask LoRA | 318 | 94.93% | 94.92% |
| Saliency mask LoRA | 319 | 95.22% | 95.22% |

Saliency masking gets one more future image right than random masking, but three fewer than ordinary LoRA. The 0.30-point macro-F1 difference over random masking is too small to establish a stable benefit from this single seed and 335-image future window. The historical model scored 87.62% macro F1 on the earlier 1980–1989 selection sample, and 93.13% on the future sample. Thus this particular split does not show a harmful temporal performance decline, even though the images are chronologically separated. It cannot support a claim of drift *correction* without a separately demonstrated harmful shift.

All four prediction files use the same future loader order; year and true class match row for row. Random and saliency predictions disagree on 11 of 335 rows. The source script and `config.json` contain exact sample limits, seeds and training hyperparameters.

## Paired injected-shift screen on future images

`injected-drift-summary.json` evaluates the saved models on the **same** 335 future images after three fixed image transforms. It measures frozen historical DINO patch-feature MMD with 499 within-pair permutations and a pair-grouped five-fold domain classifier. The random clean split-half control gives MMD² 0.00457, p = 0.462, AUC = 0.488.

The server evaluator `src/fyp_drift/evaluation/yearbook_injected_drift.py` had SHA-256 `37ba54cd50a34a7c16d204b5f450579871ebf7deba18b65148914ad54c2ae9aa` at run time. The seed-42/43/44 summary SHA-256 hashes are `8b7dc574ec84edbbffb1b6b1dbcf938c74ccf5971b4f9039ef3d378233230719`, `7b2a86a09acb6e4d36d26768ce87c9d5c11daa5510ad135e77981b688b1626c0`, and `29ea813ca2c1a82d3a3f73c095ad5cee8d6e39d6ada52881774b1886f88ca01e`.

| Paired shift | MMD² | p | Domain AUC | Historical accuracy drop | Ordinary LoRA macro F1 | Random-mask LoRA | Saliency-mask LoRA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Blur, 17-pixel kernel | 0.0264 | 0.002 | 0.998 | 0.29 pt | 96.12% | 94.63% | 94.33% |
| Center occlusion, one-third square | 0.1015 | 0.002 | 1.000 | 7.16 pt | 89.20% | 91.64% | **93.13%** |
| Gaussian noise, normalized std 0.6 | 0.1727 | 0.002 | 1.000 | 8.35 pt | 85.45% | 89.20% | 85.18% |

The center-occlusion result is a promising **exploratory** mechanism signal: saliency-mask adaptation gains 1.49 macro-F1 points over the equal-area random mask and 3.93 over plain LoRA. The other two shifts do not support a general robustness claim. The occlusion type was identified by this screen rather than fixed before seeing results, so it needs a new seed and preferably a new dataset/window. Synthetic occlusion is an injection into later dated images; its magnitude is not evidence that the original Yearbook stream underwent the same corruption.

### Seed repetition (42, 43, 44)

The same protocol was repeated without changing the shift or training settings. Each seed used all 335 future images. `injected-drift-seed43-summary.json` and `injected-drift-seed44-summary.json` are the additional server summaries; their checkpoints remain on the server.

| Seed | Historical occlusion accuracy drop | Occlusion MMD p | Ordinary LoRA F1 | Random-mask F1 | Saliency-mask F1 | Saliency − random |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 7.16 pt | 0.002 | 89.20% | 91.64% | 93.13% | +1.49 pt |
| 43 | 4.78 pt | 0.002 | 89.53% | 90.43% | 90.73% | +0.30 pt |
| 44 | 5.08 pt | 0.002 | 91.93% | 90.74% | 91.64% | +0.90 pt |

Mean saliency-minus-random is +0.90 macro-F1 points across the three seeds. Saliency exceeds random in all three, but exceeds ordinary LoRA in only two. This is a small exploratory result on a shared 335-image test set; three seeds do not establish a population-level advantage. On clean images, saliency trails ordinary LoRA by 0.90–1.50 points. On blur it trails random by 0.30–0.91 points, and on noise it trails random by up to 5.84 points. A follow-up on a different dated dataset with a frozen protocol is required.
