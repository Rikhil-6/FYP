# Saliency guided LoRA drift correction protocol

## Question and scope

Does a fixed historical class-evidence saliency map improve low-rank adaptation under acquisition shift beyond ordinary LoRA and a spatially uniform consistency loss? The saliency term is applied during adaptation. At deployment, all variants use the same patch-only ViT classifier; the map is not required for inference and no router is involved.

This is a test of correction after a labelled 2010 adaptation sample becomes available. The project has not established correction of naturally occurring year-to-year drift. Its controlled image transformations are synthetic acquisition stressors on a temporal CLEAR stream.

## Temporal split and model control

| Stage | Data | Permitted use |
| --- | --- | --- |
| Historical training | 2004–2008 train-source buckets | Train the common uniform patch-only DINOv2 model. |
| Historical selection | 2009 | Select the frozen starting checkpoint. |
| Adaptation | 2010 train-source bucket | Train LoRA and classifier on the same labelled clean/shifted image pairs in every arm. The clean view exists because the shift is synthetically generated; this assumption does not hold for arbitrary real-world drift. |
| Updated selection | A class-stratified half of 2011 train-source | Select epoch and any development hyperparameters without using future labels. The other half is reserved and not used by this classifier experiment. |
| Frozen evaluation | 2012–2014 test-source buckets | Report clean and each named shift on identical image IDs. The 2014 severity-three mixture is a stress test of known source families. |

CLEAR's year labels are upload-stream buckets. The pilot uses only seed 42 and limits adaptation, selection, and evaluation samples. Those scores are exploratory. The 2014 conditions were examined before this method was proposed, so even a later full run on them remains a follow-up, not a fresh blind test.

## Arms and mechanism

Every arm starts from the same historical checkpoint, trains rank-four query/value LoRA in the last three ViT blocks plus the same classifier, and sees the same paired views, classification losses, optimizer settings, epoch budget, and sample IDs. The arms are:

1. Ordinary LoRA with classification losses on both views.
2. LoRA plus uniform patch-feature consistency.
3. LoRA plus consistency weighted by a shuffled saliency map.
4. LoRA plus consistency weighted by the true-class top-20% patch contributions of the frozen historical classifier on the clean *training* view.

The comparison of arm 4 with arm 2 asks whether spatial weighting helps beyond consistency itself. Arm 3 tests whether any spatially sparse weighting would suffice. The historical teacher is frozen and labels are used only in the adaptation stream. The objective encourages the adapted features at selected patches to remain stable between paired views; it does not prove the selected patches depict the class object. A separate map audit is required for that interpretation.

## Evaluation and interpretation

For each frozen checkpoint, use the same image IDs in the clean and shifted conditions. Report macro F1, accuracy, shift penalty relative to that arm's same-year clean score, and paired image uncertainty. Show source-specific results and the between-seed spread for seeds 42, 43, and 44. A future experiment should hold out one source family from 2010 augmentation and evaluate it unchanged in later buckets; the current pilot does not do that. Geometric transforms need spatial alignment before patchwise consistency is applied to their paired features.

Call the result a saliency-specific benefit only if the saliency arm repeatedly exceeds ordinary LoRA, uniform consistency, and shuffled-map consistency under a declared shift without a material clean-score loss. If all arms tie, or saliency and shuffled maps tie, report no identified contribution from the map. Do not select a favorable scenario or consistency weight using 2012–2014 labels.

The pilot implementation is `src/fyp_drift/training/saliency_lora_pilot.py`. The first 600-image, one-epoch run used a conservative LoRA learning rate and produced identical predictions across arms. A second 1,000-image, two-epoch run uses a corrected LoRA learning rate and records the separate loss terms. Both pilots are retained as exploratory diagnostics.
