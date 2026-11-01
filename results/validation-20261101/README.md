# Seed 42 patch-weight intervention, 1 November 2026

This is a fixed-checkpoint inference diagnostic on the completed shared-checkpoint `patch_only` factorial. It does not train a new model or constitute an independent test: the 2014 CLEAR conditions have already been inspected. The server was `ubuntu@rik.nus.cloud` (Tesla T4); the run used the isolated `/home/ubuntu/FYP-validation-20261031` snapshot and the checkpoint `runs/patch-only-20261031/siamese/updated-patch-only-saliency-aux-seed42/models/best.pt`.

The updated classifier consumed the same backbone tokens and classifier weights under four patch-weight choices: learned, uniform, one fixed permutation, and inverted raw sigmoid gates. The inversion is `1 - gate`, renormalized across patches. This intervention checks whether the learned weights affect the trained classifier's predictions. It cannot by itself show that saliency-guided training improves generalization.

| 2014 condition, 4,999 images | Learned macro F1 | Uniform | Permuted | Inverted | Top-1 flip rate: uniform / permuted / inverted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 95.05% | 95.03% | 95.04% | 95.06% | 0.06% / 0.08% / 0.18% |
| Evidence-ranked mixture | 81.81% | 81.87% | 81.91% | 81.91% | 0.14% / 0.14% / 0.34% |

The saved full-arm 2014 mixture comparison also gives 81.81% macro F1 for the uniform updated model, 81.74% for the gated model without auxiliary losses, and 81.81% for the gated model with auxiliary losses. These are different trained arms; the weight-swap rows above hold one gated checkpoint fixed. In this seed, neither comparison shows a measurable saliency accuracy gain on the mixture. The low flip rates suggest that patch-weight content has little influence on this checkpoint's top-1 decisions, although logit changes occur. Repeat the matched arms and intervention over the declared seeds and conditions before making a general claim.

Command (run from the isolated snapshot):

```bash
PYTHONPATH=src .venv/bin/python -m fyp_drift.evaluation.patch_mechanism \
  --run-dir runs/patch-only-20261031/siamese/updated-patch-only-saliency-aux-seed42 \
  --data-root data --year 2014 --batch-size 16 --role updated \
  --scenarios clean realistic_mix \
  --output runs/patch-only-20261031/patch-mechanism-inverted-full-2014.json
```

The machine-readable report is [patch-mechanism-inverted-full-2014.json](patch-mechanism-inverted-full-2014.json). The new local intervention test passed via `python -m unittest tests.test_patch_mechanism -q` (4 tests). The server's earlier four-arm script exited 1 only after training, when router-feature replay found 2 decision mismatches among 164,945 rounded logged features. That router issue is separate from this saliency diagnostic.

SHA-256 provenance:

- Model source: `8260e6c59410c285b489a21e233aa65a3ccb7d3c5266494287faaf38c1decc34`
- Evaluator source: `fab2033c035b8b1343f31129d8420d9fd5ebc35ce72ef1be7f1c706510e88c2b`
- Source-run config: `b287d20a27555cbd4b5d1e59e4de57b29807dca4a1a53b522823a10a644a708d`
- Updated checkpoint: `1d145862b3c20dc378963e8eb88543dbe2fd3881a5f3fb89153f3496230043b5`
- Output JSON: `b18e5fe9f9a3dfcbe608b3a3c2b931a10b8b84dc7c629c69ca80d63d9de71f19`
