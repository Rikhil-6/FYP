# FYP execution plan: test the saliency mechanism

## Material Passport

- Origin: refinement of the supplied `FYP-execution-plan.md` against the current checkout and the 31 October 2026 validation records
- Mode: experiment plan
- Date: 1 November 2026
- Verification status: plan updated from recorded results; the new server pilot and factorial results are not verified here
- Version: mechanism_plan_v2

This plan addresses the central question: **when image capture or context changes while the class still means the same thing, does learned patch weighting preserve task-relevant evidence better than ordinary updating?** The proposed benefit is specific. If the model can put more weight on stable object evidence and less on a changed background or local nuisance, its updated predictions may degrade less under that shift. A visible saliency map alone does not establish this benefit. The map must affect predictions, and a matched model without learned weighting must perform worse on the relevant conditions. Routing is a secondary extension after the classifier mechanism and utility have been established.

## Current evidence and claim boundary

The corrected 2011 split has already been used for a seed 42 server validation. The updated saliency model scored 80.78% macro F1 on the 2014 evidence-ranked mixture, against 80.81% without auxiliary saliency losses; routed scores were 79.75% and 79.60%. A corrected conventional fine-tuning arm scored 81.55% on that mixture. It beat the saliency update in nine of eleven 2014 conditions, tied in one, and lost in one. That strategy comparison starts from different historical heads, so it does not isolate the patch gate's effect. The fixed-checkpoint router-feature ablation has also been completed: at equal 50% disagreement coverage, confidence-only scored 79.98% and combined scored 79.74%. These are single-seed, already inspected results, not a fresh confirmatory test. See `paper/argument_audit_20261031.md` and `results/validation-20261031/` for the records.

The earlier plan's policy-replay implementation task is therefore complete and must not be presented as future work. The **seed 42 shared-checkpoint, patch-only factorial has now trained all four model arms**: on the 2014 evidence-ranked mixture, uniform updating and gated updating with auxiliary losses both score 81.81% macro F1; gating without auxiliary losses scores 81.74%. The script exited after training because secondary router replay found two mismatches among 164,945 rounded future-feature rows. A full fixed-checkpoint intervention on 4,999 clean and 4,999 mixture images found that uniform, permuted, and inverted weights changed only 0.14%, 0.14%, and 0.34% of mixture top-1 predictions. None lowered mixture macro F1. See `results/validation-20261101/`. This single-seed, already inspected evidence does not show a saliency accuracy advantage; seeds 43 and 44 and source-specific checks remain open.

## Causal chain and tests

| Link in the argument | Required observation | Competing explanation or boundary |
| --- | --- | --- |
| The incoming images change | Detect distribution change on fixed-size, label-free image samples; separately measure task loss | Detectable feature change need not damage classification. Clean temporal change has not yet been detected reliably. |
| The change disrupts context more than the class evidence | Specify the nuisance, severity, and stable label assumption before scoring | Strong blur and compression may destroy the object evidence itself. |
| Learned weighting influences the classifier | Replace learned weights with uniform and permuted weights at fixed checkpoint; compare logits and predictions. Train a patch-only head that removes the direct class-token-to-head path. | Visually plausible maps and deletion panels do not prove prediction dependence. Replacement can create an off-distribution internal state. |
| Saliency improves adaptation | Compare uniform, gated without auxiliary losses, and gated with auxiliary losses from the same historical checkpoint, data, seed, tuning budget, and selection split | Robustness augmentation, model capacity, or checkpoint differences could explain an apparent gain. |
| Routing adds decision value | Replay confidence-only, map-only, and combined policies on the same frozen model pair and calibration/evaluation IDs; compare with always-updated at equal route coverage and declared error costs | Better explanations or fewer harmful routes alone do not imply better macro F1 or net utility. |

The likely benefit should be tested first on localized context or framing changes, including viewpoint and modest occlusion. Report global degradations such as blur, noise, and JPEG compression as separate stressors. A gain under those conditions may be useful, but should not automatically be attributed to spatial saliency. The method does not address changed label definitions, unseen classes, severe loss of object information, or annotation errors.

## Study sequence

1. **Freeze the next protocol.** Record the server checkout commit, hashes of uncommitted helper scripts, lockfile and CLEAR manifest, seeds, exact commands, named conditions, metrics, and selection rules. Preserve the 2004–2008 train / 2009 select / 2010 adapt / disjoint 2011 select and calibrate / 2012–2014 evaluate chronology. Report all eleven declared conditions, not just the evidence-ranked mixture. Label all 2012–2014 analyses as corrected-protocol replications or exploratory follow-ups because those data have been inspected.
2. **Reconcile the completed seed 42 server run.** The four training arms and full clean/mixture weight-swap diagnostic are complete. Verify the common historical checkpoint hash, uniform gate initialization, disjoint 2011 IDs, identical future IDs, and all source-specific scores from saved artifacts. The earlier script's dependence on another progress log and three-epoch historical budget must be reviewed against the frozen protocol before repeating it. The router replay's two rounded-feature mismatches are a secondary logging issue; preserve the failure record.
3. **Complete the saliency evidence table.** Run the fixed-checkpoint learned/uniform/permuted/inverted intervention for every declared 2014 source on the seed 42 checkpoint. Save prediction flips, logit changes, accuracy, and macro F1. Compare those results with the separately trained uniform and gated arms, keeping the fixed-checkpoint and training comparisons distinct. Record checkpoint, configuration, source-code, output, and sample-ID hashes.
4. **Repeat matched seeds.** Repeat the frozen factorial for seeds 43 and 44. Keep the same starting checkpoint *within each seed* and the same adaptation budget across arms. Do not tune the gate, thresholds, or chosen corruption subset on future labels. Report per-seed paired image intervals and between-seed spread separately.
5. **Revise the paper around the saliency result.** Put the research problem, hypothesized patch-weight mechanism, fixed-checkpoint intervention, and shared-checkpoint comparison before the router. Show the complete condition table, negative results, and provenance. Keep archived shared-calibration scores separate from corrected runs. The supplied `FYP report framing plan.docx` is a writing and test guide, but its positive 1.20-point comparison mixes split protocols and is superseded by the corrected single-seed conventional result; do not carry that figure into a current headline claim.
6. **Test routing as a secondary extension.** For each frozen model pair, compare always-historical, always-updated, confidence-only, map-only, and combined routing. Match route coverage or use a predeclared streaming threshold. Report macro F1, coverage, helpful/harmful/missed switches, classwise effects, and a prespecified cost sensitivity. A quota set from the full future batch is a transductive diagnostic, not a deployable streaming policy.

## Run integrity checks

The saved seed 42 factorial is ready for interpretation only after confirming that all four model paths finished on the intended hardware, the three updated arms share the exact historical starting weights, 2011 selection and calibration IDs do not overlap, evaluation IDs match across arms, output files parse, and runtime/storage estimates come from logs. If any check fails, fix the cause, version the protocol, and rerun the affected stage before repeating the factorial. The 500-image pilot accuracy was diagnostic and was not used to choose an arm or condition.

## Saliency code and test checks

- The fixed-checkpoint intervention now includes learned, uniform, permuted, and **inverted raw-gate** weights. Inversion uses `1 - sigmoid(gate_logit)` followed by normalization, matching the model's complementary-patch definition. Check prediction flips, true-class logit changes, accuracy, and macro F1 by condition. An intervention tests model dependence on the weights; it does not alone prove a training benefit.
- Test that all four interventions use the same backbone tokens and classifier weights, that inverse weights are normalized correctly, and that a direct class-token-only head masks changes to the pooled patch representation. Keep a trained patch-only comparison to address that bypass.
- In the server pilot, assert that uniform and gated patch-only models start with identical logits from the shared historical uniform checkpoint, and that changing the saved checkpoint or 2011 split fails early on resume. Compare the complete output schemas and sample IDs before interpreting any score.
- The report-framing document's disagreement-only router fit and random-coverage checks remain secondary. They cannot repair a missing saliency classifier effect and should follow the shared-checkpoint result.

## Interpretation rule

Claim a **bounded saliency-guided adaptation benefit** only if the shared-checkpoint comparison shows a repeatable gain over uniform updating in predeclared relevant conditions, the learned gate materially changes predictions, and the gain survives the no-auxiliary comparison. If the gate changes predictions but does not improve task performance, report a mechanism without a demonstrated utility gain. If the map does not materially affect predictions, describe it as an inspectable signal rather than a proven cause of robustness. Claim routing benefit only where it beats always-updated or a declared cost criterion; its audit trail is a separate contribution.

The 1 November 2026 server pilot and full clean/mixture fixed-checkpoint diagnostic are recorded in `results/validation-20261101/`. The multi-seed factorial and remaining source-specific interventions are still proposed work.
