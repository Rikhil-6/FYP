# External data for a balanced saliency claim

Select datasets before inspecting new outcomes. Report every named domain and
seed, including losses. The existing frozen-head screen finds a small CIFAR-10.1
gain (+0.30 ± 0.11 macro-F1 points across three seeds), but Yearbook (-0.33,
one seed) and Camelyon17 (-0.07 ± 0.12, three seeds) do not corroborate a
general advantage. These screen results test a head over frozen features, not
the full adapted router.

| Dataset and source | What it tests | Limitation and planned use |
| --- | --- | --- |
| [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) and [CIFAR-10.1 v6](https://github.com/modestyachts/CIFAR-10.1) | Independent recollection with the same ten classes; already gives the modest positive head result | Recollection is not a longitudinal update stream. Preserve all three existing seeds and report the small effect beside clean accuracy. |
| [CIFAR-10-C](https://zenodo.org/records/2535967) | Fixed common-corruption test over named sources and five severities | Synthetic corruptions are not calendar drift. Evaluate every source/severity with the same frozen checkpoint and no test-label tuning; report gain and harm counts. |
| [CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html) and [CIFAR-100-C](https://zenodo.org/records/3555552) | Hundred-class stress test closer to CLEAR's label count | Still 32-pixel images and synthetic shift. Upscaling to the DINOv2 input size may blur patch-level interpretation; disclose it. |
| [Wild-Time Yearbook](https://wild-time.github.io/) | A genuine temporal counterexample already in the paper | One existing seed and frozen heads only. Repeat with matched seeds before inferring a temporal advantage. |
| [CIFAR-10-Warehouse](https://github.com/sxzrt/CIFAR-10-W) | Broader domain coverage if resources allow | Search-engine and generated-image domains can introduce selection and label-quality differences. Preselect domains without looking at model results. |

## Minimum fair protocol

1. On each dataset, fit uniform and saliency heads with the same backbone,
   images, split, seeds, epochs, optimiser schedule, and checkpoint rule.
2. Keep separate claims for the frozen-head screen and a full historical,
   updated, calibrated-router experiment. CIFAR-10.1 and CIFAR-C do not by
   themselves support real temporal-drift correction.
3. Report macro F1, clean score, per-domain scores, seed spread, and paired
   uncertainty. For routing also report always-historical, always-updated,
   confidence-only, saliency-only, combined, coverage, harmful switches, and
   missed beneficial updates.
4. Advance the narrow claim only if a predeclared aggregate improves without
   unacceptable clean degradation; preserve contrary domains in the same table.

The downloads and protocol should cite the original dataset sources above.
The corrected CLEAR conventional arm in `deploy/run_corrected_finetune_20261031.sh`
is the priority before adding more external experiments.
