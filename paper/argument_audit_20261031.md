# Argument audit — 31 October 2026

Read against `saliency_guided_vit_drift_routing_revised.docx`, the code, and the
corrected single-seed validation. This is a review, not replacement thesis prose.
The corrected conventional arm has completed. Its 2014 mixture result reverses
the archived saliency-versus-conventional comparison; the shared-checkpoint
patch-only factorial remains queued.

## What the evidence currently supports

The project implements a reproducible historical-versus-updated ViT choice with
an inspectable logistic rule and per-image traces. On a severe engineered CLEAR
mixture, the archived strategy table favours updating, while the router retains
most of that archived gain but trails always-updated inference. A separate
corrected-split single-seed run gives a smaller mixture gain (79.33% historical
saliency, 80.78% updated, 79.75% routed). Its auxiliary-loss effect is +0.15
routed points and -0.03 updated points against the no-auxiliary control. At 50%
disagreement coverage, confidence-only routing scores 79.98% versus 79.74%
for the combined gate. The existing external positive result is a small
CIFAR-10.1 frozen-head effect; Yearbook and Camelyon do not corroborate a
general head advantage.

The corrected conventional update scores 81.55% macro F1 on the 2014
evidence-ranked mixture, against 80.78% for updated saliency and 80.81% for its
no-auxiliary control. Corrected saliency is lower than conventional in nine of
eleven 2014 conditions, tied in one, and higher in one; clean 2014 is tied at
94.93%. This reverses the archived 1.20-point mixture advantage, but different
historical heads and one seed prevent a causal saliency-penalty conclusion.

The third experiment now has a separate cost-sensitivity check for the same
fixed model pair. On the 2014 mixture there are 379 available beneficial and
198 harmful replacements. If each harm costs one missed benefit, always-updated
leads; at twice that cost, confidence-only leads; at three times, always-
historical leads. The combined gate does not lead at those values, even after
matching route coverage. This makes the router's auditability a concrete
contribution while leaving its incremental decision benefit unproven.

A preliminary fixed-checkpoint intervention on 500 class-balanced 2014 images
shows why the mechanism needs a stricter control. On the evidence-ranked
mixture, learned, uniform, and once-permuted patch weights score 78.57%,
78.35%, and 78.60% macro F1 with the same updated checkpoint. Masking the patch
summary gives 77.44%; masking the direct class-token input gives 59.57%.
Those masks are outside the trained input distribution, and this is one
permutation and one subsample. They indicate a substantial direct class-token
path and motivate the queued full intervention and shared-checkpoint
`patch_only` comparison; they do not establish a stable saliency effect.

## Research-question chain

| Question | What the paper establishes | Where the inference stops |
| --- | --- | --- |
| RQ1: detectability from limited unlabelled images | A frozen DINOv2 embedding, paired or unpaired MMD test as appropriate, and grouped domain AUC detect the declared controlled shifts. A new label-free random-sample audit detects motion blur and the evidence-ranked mixture at 100, 300, and 500 images per group (all p = 0.002); clean 2009 versus 2014 is not detected (p = 0.268, 0.058, 0.136), and the disjoint 2014 null is not detected. | These are tests on representations derived from raw images, not a universal raw-pixel drift detector. One nested sample draw and one representation cannot establish power across natural changes. Detection does not establish task harm, and the earlier 300-image audit balanced classes using labels at sampling time. |
| RQ2: added value of saliency-guided updating | The archived table shows a 1.20-point 2014 mixture advantage, but the corrected conventional arm is 0.77 points ahead of the corrected saliency update; the auxiliary-loss comparison is nearly tied. | Different historical heads and one seed prevent attributing either direction to saliency. The shared-checkpoint patch-only factorial and fixed-checkpoint intervention are needed for a component claim. |
| RQ3: useful and safe routing as a separate experiment | The implementation replays decisions and exposes branch signals and helpful, harmful, and missed switches. The archived router retains 88.3% of the updated gain. | It trails always-updated inference, and confidence-only scores higher in the corrected equal-coverage feature test. Routing value needs a cost-sensitive, matched policy comparison; it cannot serve as evidence that saliency improves the updated classifier. |

## Priority objections and the evidence needed

| Priority | Argument step | Logical gap or rival explanation | Decisive next check |
| --- | --- | --- | --- |
| 1 | RQ2; Sections 3, 4.5, 5.1, 7.4, 8.1 | The archived 1.20-point saliency-versus-conventional mixture gap reverses in the corrected split: conventional is 0.77 points ahead. The corrected saliency/no-auxiliary effect is small and mixed. Both strategy comparisons use different historical heads, so neither identifies a causal saliency effect. | Complete the queued shared-checkpoint patch-only factorial, repeat matched seeds, and compare uniform/gated heads with and without auxiliary loss. |
| 1 | RQ3; Sections 3.2, 6.1, 7.2, 7.4, 8.1 | A fall-back option is real, but archived routing trails always-updated inference in all 11 conditions. It avoids 27 harmful switches while missing 50 beneficial ones on the mixture. The corrected fixed-model feature test does not show combined features beating confidence-only at equal 50% coverage. Calling routing safer needs a loss function, not only a trace. | Predeclare costs for harmful and missed updates; compare expected utility and risk-coverage curves with always-updated and confidence-only policies at equal coverage and latency. |
| 1 | RQ1 and temporal claim; Sections 4.2–4.3, 5.1, 8.3 | Year, synthetic severity, and mixture component count rise together. The sampled clean-year frozen-feature test did not detect separation. The largest positive evidence is therefore compound-corruption robustness, not demonstrated correction of natural time drift. | Cross year with fixed corruption severity/source count, include clean-year comparisons with larger samples, and model a year × corruption interaction. Add another native temporal dataset. |
| 1 | Mechanism; Sections 3.1, 5.4.1, 6.2, 7.2 | The classifier concatenates an unrestricted class token with the gated patch summary. The class token may carry most of the decision. The complementary-patch loss leaves it unchanged. A visually distinct map and deletion margin do not establish that the learned gate materially changes the predicted class. | At inference, replace learned patch weights with uniform and shuffled weights while holding the class token and head fixed; measure logit, accuracy, and routing changes. Compare with a class-token-only head. |
| 2 | Calibration; Sections 3.2, 4.6, 7.3 | The corrected 2011 selection split protects the checkpoint choice, but logistic coefficients, saliency floor, and threshold are fitted/chosen on the same routing-calibration labels. Its reported 2011 score is optimistic. The balanced logistic output is not a validated probability. | Split 2011 again or cross-fit policy parameters and threshold; report held-out reliability, Brier/log loss, route coverage, and future outcomes. Keep the deployment threshold frozen. |
| 2 | Source selection; Sections 4.2–4.3 | Ranking corruptions by MMD and domain AUC selects feature separation, not plausible prevalence or improved task performance. It preferentially creates a severe stress test and cannot identify whether gain comes from saliency or ordinary robustness training. | Report all predeclared source/severity cells and the selection rule; add a severity-matched random mixture and a corruption-augmentation baseline. Do not interpret the top-three mixture as field frequency. |
| 2 | Weak-class argument; Sections 5.4–5.4.1 | Q1/Q4 are defined with historical F1 on the same 2014 evaluation labels. Low groups have more headroom and regression-to-the-mean risk. This does not show the gate detects weak classes before their outcomes are known. | Freeze class-difficulty ranks using pre-2012 data, then evaluate later classwise changes and confidence intervals. |
| 2 | Faithfulness; Section 6 | True-class deletion uses labels unavailable to deployment and creates out-of-distribution images. There is no random-patch or matched-area control. Comparing deletion scores after grouping by which branch was correct also conditions on the outcome and may exaggerate the apparent separation. The `saliency_only` policy includes class-evidence gain computed from softmax outputs, so the name is not a pure spatial-map intervention. | Add random/bottom/high patch deletion or insertion at equal area, robustness to replacement baseline, and a map-only gate (`saliency_js`) beside evidence-only and confidence-only policies. Report deletion effects before and after correctness grouping. |
| 2 | Statistical precision; Sections 2.4, 4.7, 5, 7, 8 | One CLEAR seed and many inspected conditions make small effects, including +0.15 routed points, fragile. Image-level bootstrap alone conditions on the trained checkpoints and misses training variance. Post-result 2-point/75% checks are descriptive. | Report all conditions, paired image intervals, and at least three matched training seeds; distinguish within-run uncertainty from between-seed variation and avoid choosing a favourable condition after inspection. |
| 3 | External validity; Sections 4.4, 5.5, 7.4 | CIFAR-10.1 tests an independently recollected 10-class set, not temporal updating; CIFAR-C tests synthetic corruption; the external code tests frozen heads, not the full router. Yearbook's negative single run and Camelyon's near-null result belong beside the CIFAR positive result. | Predeclare the datasets/domains in `docs/external_dataset_evidence_plan.md`, repeat Yearbook, and run a full historical/update/calibration/test design only where native splits permit it. |
| 3 | Framing and novelty; Introduction, Sections 2.2–2.3, 8.2 | The Siamese label describes two versions of one ViT, not a learned Siamese metric architecture. The credible contribution is an integrated, inspectable update workflow; saliency's incremental predictive value and a unique safety advantage remain unproven. | Present the design as an implemented research system and state the unresolved mechanism tests before stronger novelty or deployment claims. |

## Cross-section consistency to resolve after the running arm

1. The abstract and Figure 4 foreground archived 82.75/82.36 mixture scores,
   while Section 7.4 reports corrected 80.78/79.75 scores against a different
   historical branch. Keep the denominators and protocol labels adjacent
   whenever comparing them. Do not average the two snapshots.
2. Sections 4.5 and 5.1 describe conventional fine-tuning as the main control.
   The corrected result is now in Section 7.4 with its 11-condition direction
   count. Add a full condition table and paired uncertainty in the final
   submission, not only the severe-mixture comparison.
3. The archived Figure 7 still labels the logistic output “route probability.”
   The caption explains that this is an implementation field name, not a tested
   reliability claim. Regenerate the figure with “routing score” if its archived
   per-image data become available.
4. The documented 2014 mixture has 4,999 images and 49–50 examples per class.
   Do not adjust a class count for symmetry; report the observed support.
5. Sections 4.5 and 5.1 discuss a fixed-model feature ablation as future work,
   but Section 7.4 already reports it. Point those earlier passages to the
   later negative result. Section 8 should describe it as completed.
6. The abstract, Figure 4, and concluding language still give the archived
   auxiliary-loss advantage more weight than the corrected near-zero effect.
   Both can be reported, but the strongest end-to-end claim should follow the
   matched corrected protocol. Name each historical checkpoint when comparing
   conventional fine-tuning runs so a reader can distinguish archived and
   corrected controls.
7. Helpful/harmful switch counts do not directly explain a macro-F1 change:
   macro F1 is a nonlinear aggregate of each class's precision and recall.
   Show classwise confusion changes or an explicit per-image cost beside those
   counts.

## Decision rule for the final FYP argument

If the shared-checkpoint factorial and multi-seed saliency arms show a stable
advantage under predeclared conditions, the paper can claim a bounded
saliency-guided strategy benefit. If not, the defensible contribution remains
the transparent update-and-audit system plus a careful account of when its
saliency features do and do not help. Either result is useful if all negative
conditions and limitations stay in view.
