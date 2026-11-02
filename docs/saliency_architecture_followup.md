# Saliency as a drift-correction mechanism

The question is whether an adaptation method uses saliency to preserve task-relevant visual evidence when acquisition conditions change, beyond what ordinary fine-tuning or a generic augmentation would do. The router is a downstream option and is not needed to test this mechanism.

## Literature and design choices

- [Ma et al., *Rectify ViT Shortcut Learning by Visual Saliency*](https://arxiv.org/abs/2206.08567) restrict patch self-attention to salient patches while retaining a global residual. Our existing post-backbone saliency gate is weaker: it changes pooling after representations have formed. The learned-gate weight-swap test found virtually no causal contribution, so a new attempt should intervene **inside** representation learning, with a global path retained.
- [Ismail et al., *Improving Deep Learning Interpretability by Saliency Guided Training*](https://arxiv.org/abs/2111.14338) mask low-saliency features and encourage output agreement. This motivates the implemented saliency-mask LoRA training arm. The current pilot uses supervised CE on the masked view; an output-consistency term against the unmasked prediction is a distinct next ablation. The literature reports interpretability while preserving performance, not a guarantee of drift correction.
- [Zheng et al., *Instance-Warp*](https://arxiv.org/abs/2403.12712) use saliency to emphasize foreground objects during domain adaptation. This suggests a spatially local adaptation mechanism when camera background, lighting or hospital acquisition changes. For classification, a bounded patch reweighting or saliency-aware crop can be compared against a random crop with the same area and training budget.

## Test order

1. Verify the target window has distribution change and **historical-model harm**: clean split-half null, MMD² and permutation p, domain-classifier AUC, and paired prediction loss. Separate natural chronology from an injected perturbation.
2. Compare ordinary LoRA, random equal-area masks and saliency masks with identical backbone, trainable rank, IDs, seeds, epochs and selection rule. Keep all future labels out of adaptation and model selection. Report clean and shifted future performance together.
3. If masking consistently helps, test output-consistency training and a patch-level residual intervention separately. For a residual, preserve the global path and compare learned saliency, shuffled maps and uniform maps while holding parameter count fixed.
4. Repeat on a real chronological time × camera or time × acquisition dataset. A gain on synthetic occlusion alone only establishes mechanism sensitivity.

The current Yearbook result is selective: saliency-mask LoRA beats random masking under center occlusion in seeds 42–44 by 0.30–1.49 macro-F1 points (mean +0.90), but trails it under blur and usually noise, and trails ordinary LoRA on clean images. It beats ordinary LoRA under center occlusion in only two of three seeds. Thus the supported statement is an exploratory robustness signal for **one** injected occlusion condition on a later dated image set, not correction of the natural Yearbook drift. The exact numbers and drift tests are in `results/validation-20261102/yearbook-saliency-mask/`.
