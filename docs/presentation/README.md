# Presentation slides

**Saliency-Guided Data Drift Mitigation on Vision Transformers**
Rikhil Singh, supervised by Prof Markus Kirchberg (13 November 2026).

[Slides (PDF, 57 pages)](Saliency-Guided_Data_Drift_Mitigation_ViT_slides.pdf)

The deck presents the final-year project in the same order as the paper. Numbers in the deck are the paper's
three-seed results; the same tables are in the [main README](../../README.md) and can be recomputed from the code
(see the [paper guide](../paper_guide.md)).

## What the deck argues

- Natural fMoW temporal drift is detectable but weak, so a controlled harm (realistic cloud cover) is injected.
  It costs the frozen model about 7 macro-F1 points (49.1 to 41.9).
- Cheap updates recover most of the loss. Saliency-gated heads train in about a minute with the backbone untouched;
  six-block LoRA is the accuracy leader (about 36 minutes, 75.5M backbone weights changed).
- Saliency does not buy accuracy: gates match learned attention and their shuffled-map twins. It buys exact,
  object-grounded, more stable explanations and a decision log that can be audited.
- A router that keeps the old answer unless the update is clearly better is logged per image; confidence alone
  carries the decision, and the saliency signals serve traceability.

## Where things are

| Slides | Section | Content |
|---|---|---|
| 1-8 | Motivation | Why look inside the model, what saliency is, the drift problem, research questions RQ1-RQ4 |
| 9-12 | Data | fMoW, splits, drift versus harm, the injected cloud |
| 13-22 | System | The saliency gates, hybrid LoRA plus soft gate, design choices |
| 23-28 | Results | Metrics, update comparison, gate performance, cloud over the object, faithfulness |
| 29-33 | Discussion | Where saliency earns its place, grounding, what it does and does not change |
| 34-39 | Router | Overcorrection, certainty versus positionality, stored policy, router results |
| 40-47 | Closing | Conclusions, research questions answered, rollout, limitations, future work |
| 48-53 | Appendix | Full results table, LoRA lead under cloud, twins, overcorrection examples, run lifecycle |
| 54-57 | References | Numbered by first appearance in the deck |

Page ranges refer to the 57-page PDF; each page header names its section.

## Notes that are not printed on the slides

- Numbers are macro-F1 (%) on the 2016-17 test images, models trained on clean images; three-seed means
  (42, 43, 44) except the frozen baseline, which is a single run. Shaded rows in the appendix table are the
  saliency-guided models; bold marks the best value in a column.
- Training times are for seed 42 on one T4. "Backbone weights changed" means weights changed inside DINOv2-L (304M).
- The overcorrection examples (page 52) are hand-picked from seed 42. The hybrid pictured there is rank 1 with
  47k trainable parameters; the hybrid scored in the results is rank 2 with 84k, so the maps illustrate the design
  rather than the exact scored model.
- Gain-over-LoRA percentages (page 51) are measured against six-block rank-2 LoRA (100%).

## Limitations to keep in mind

One dataset (fMoW) and one frozen backbone (DINOv2-L/14); saliency is derived from the old model itself;
the cloud drift is designed rather than natural; absence of natural harm is not proof of no harm; there is no
shuffled-map hybrid control. The slides state these on the Limitations page.
