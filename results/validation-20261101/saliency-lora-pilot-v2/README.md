# Saliency-guided LoRA pilot on CLEAR

Run date: 1 November 2026. Hardware: Tesla T4 on the isolated FYP validation server snapshot. This is exploratory: the 2014 CLEAR conditions had been inspected before the method was designed. The source checkpoints and full per-image files remain on the server under `runs/saliency-lora-pilot-v2-20261101/` and `runs/saliency-lora-pilot-v2-full2014/`.

## Design

The common historical patch-only uniform DINOv2 checkpoint was trained on 2004–2008 and selected on 2009. Four rank-four query/value LoRA arms adapted on the same 1,000 labelled 2010 image IDs, with paired clean and synthetically shifted views. The arms were ordinary LoRA, uniform patch consistency, shuffled saliency consistency, and fixed-teacher saliency consistency. Every arm trained the same 56,932 parameters, used two epochs and the same optimizer. Epochs were selected using the same class-stratified 2011 selection partition. The 2014 test used identical 4,999 image IDs within every scenario and arm; [id-check.json](id-check.json) verifies all 19,996 year/scenario rows match.

The saliency map selected the top 20% of historical patch contributions to the known *training* class on the clean view. The teacher was frozen. No saliency map or label was used at evaluation. This is a training regularizer, not an inference-time gate.

## Full 2014 results

Macro F1, percent. The 500-image pilot scores are replaced by these complete 2014 scores.

| Condition | Ordinary LoRA | Uniform consistency | Shuffled map | Saliency map |
| --- | ---: | ---: | ---: | ---: |
| Clean | 94.58 | 94.54 | 94.54 | 94.52 |
| Viewpoint shift | 94.06 | 93.94 | 93.96 | 93.98 |
| Occlusion | 93.33 | 93.64 | 93.55 | 93.63 |
| Evidence-ranked mixture | 79.21 | 79.17 | 79.19 | 79.23 |

The saliency arm exceeds ordinary LoRA by 0.02 points on the mixture and by 0.30 points under occlusion. Uniform consistency improves occlusion by 0.31 points, so the occlusion result does not isolate a saliency effect. Saliency trails ordinary LoRA on clean and viewpoint shift. These single-seed, small differences are not evidence of a repeatable saliency-specific drift correction.

The first 600-image, one-epoch pilot used a conservative shared learning rate for LoRA and the classifier and produced identical predictions in all arms. The second pilot corrected the LoRA rate to `1e-3`, retained `5e-5` for the classifier, and used a consistency coefficient of 1.0. Both choices were made in an exploratory phase, before the full 2014 pass but after previous CLEAR results had been seen. The selection scores fell from epoch one to epoch two, so the saved best checkpoints were selected on 2011. See [training-summary.json](training-summary.json), [full-2014-summary.json](full-2014-summary.json), and [config.json](config.json).

## Interpretation and next test

The current final-pooling gate and this saliency-weighted consistency loss do not establish the desired benefit. The paper already cites [Ma et al., Rectify ViT Shortcut Learning by Visual Saliency](https://arxiv.org/abs/2206.08567), which uses salient patch selection inside attention with a global residual; [Zheng et al., Instance-Warp](https://arxiv.org/abs/2403.12712), which emphasizes foreground objects during adaptation; and [Ismail et al., Saliency Guided Training](https://arxiv.org/abs/2111.14338), which masks low-saliency inputs and trains output consistency. Those mechanisms act before or during representation learning, whereas this pilot weights a small auxiliary loss on last-layer features. A next method should be chosen from that mechanism distinction and tested against a matched non-saliency control, first on a background-shift benchmark, then on a dataset with genuine temporal or site shift. Avoid using the already inspected CLEAR mixture to choose a favorable method.

The official [Waterbirds dataset](https://github.com/kohpangwei/group_DRO#waterbirds) isolates foreground/background correlation but has no chronology. [WILDS](https://wilds.stanford.edu/datasets/) offers Camelyon17 hospital shift, iWildCam camera-location shift, and FMoW temporal satellite shift. ACDC, already discussed in this project, has matched normal/adverse scenes and segmentation labels but no chronology. The server's folders for Camelyon17, iWildCam, and FMoW are empty; the official Waterbirds archive was not reachable from the server on 1 November 2026. No external-data experiment is claimed here.

## Provenance

- Pilot module SHA-256: `0ff1d6b555cfdcf9282830c64ea2d3d24fc5eab93584e35ed1041a7f323d239b`
- Full evaluator SHA-256: `8e85521169ca26af700d3a1b35d000b695fd902ab535349ba22f14fb6b223dbe`
- Configuration SHA-256: `56559068aadef585dd966ac56056db850101fc4b3c50402fac53d410ddd75613`
- Historical checkpoint SHA-256: `168e6291a8e4170e13d4b40c8bf0854940ca48d054579e38ea8d0a40f422f8f1`
- Full summary SHA-256: `c09b58271e37d2a2506a92187fc565a2e9a2aed9b9b5dc8152f4d2f2d33c6d30`
- Paired-ID check SHA-256: `0f4c38ec3bd246264d749db01f7f17ab44f49089d3bf759fe6ad789f38d14cfe`
