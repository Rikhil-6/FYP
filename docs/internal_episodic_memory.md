# FYP Internal Episodic Memory

Last updated: 2026-10-05 (Asia/Singapore)

## Purpose

This file preserves the working context and decisions from the prototype-development session. It is an internal handoff reference, not a polished research claim. Do not add credentials, private-key contents, dataset archives, or model checkpoints to Git.

## Project identity

- Repository: `Rikhil-6/FYP`
- Active experimental branch: `Rik/transparent-router`
- Local isolated worktree: `work/transparent-router`
- Server checkout: `/home/ubuntu/FYP-fork`
- Shared server storage: `/mnt/sdd/FYP-fork`
- Dataset: CLEAR-100, using upload year as the temporal axis
- Backbone: pretrained `facebook/dinov2-small`
- Environment and commands: uv only; `.venv`, data, run, tool, and cache locations are symlinked on the server
- Server accelerator observed on 2026-10-05: Tesla T4, 14.56 GiB; PyTorch `2.14.0+cu130`

The SSH private key is ignored by Git and must remain local. GitHub credentials are not installed in the server checkout. Repository synchronization has therefore used a local Git bundle transferred over SSH.

## User requirements retained

- ViT-only models; no CNN branch.
- Data ingestion, model construction, training, evaluation, and artifact storage remain separate.
- Temporal evaluation must resemble deployment: fit on an earlier period, adapt/calibrate on later but pre-test periods, then freeze the model over a future horizon.
- XAI must affect model computation or routing rather than exist only as a post-hoc image.
- Routing must expose a clear relationship between saliency change and the update decision.
- Record accuracy, macro F1, macro IoU, routing metrics, training time, inference time, logs, checkpoints, and progress bars.
- Save best and last checkpoints separately.
- Save representative saliency images in a bounded Git-friendly report directory.
- Keep server execution easy to launch and observe through uv, shell scripts, tmux, and persistent logs.

## Current architecture

### Historical baseline ViT

`TemporalViT(method="baseline")` uses DINOv2 class and patch tokens. Patch features are pooled uniformly and concatenated with the class token before the classifier. This is the original non-saliency reference.

### Embedded-saliency XAI ViT

`TemporalViT(method="xai")` learns one scalar gate per patch token. Normalized gate values weight the patch sum used by the classifier, so the saliency map directly changes the prediction. A complementary low-saliency path supplies counterfactual class evidence. The map is a learned token gate and should not be described as raw transformer self-attention.

### Historical/newer ViT pair

The pseudo-Siamese setup copies the historical XAI checkpoint into two towers. The old tower is frozen. The newer tower is adapted on 2010 data. Training combines:

- cross-entropy classification loss;
- pairwise saliency divergence: keep maps close when the old model is correct and impose a divergence margin when it is wrong;
- counterfactual evidence loss: salient patches should provide more true-class evidence than their complement.

### Transparent router

The router uses four logged features:

1. saliency Jensen-Shannon divergence;
2. confidence gain from old to new;
3. entropy reduction;
4. counterfactual evidence gain.

Features are standardized using 2011 calibration statistics and passed through a logistic rule. The newer prediction is used only if the models disagree, saliency divergence clears an explicit floor, and route probability clears the calibrated threshold. Policy means, scales, coefficients, intercept, threshold, and saliency floor are saved in `metrics/router_policy.json`.

## Frozen chronology

| Period | Purpose |
|---|---|
| 2004–2008 | Historical training |
| 2009 | Historical checkpoint selection |
| 2010 | Newer-XAI adaptation |
| 2011 | Newer checkpoint selection and router calibration |
| 2012–2014 | Frozen future evaluation |

Evaluation conditions are clean, saturation change, Gaussian noise, partial local-mean occlusion, and their combined composition. Individual corruptions are available during drift-aware adaptation/calibration; combined corruption is held out until future testing.

## Implemented experiment profiles

### Standard profile

- 50% clean / 50% individual severity-1 corruption mixture.
- Final ViT block trainable.
- Learning rate `1e-5` for trainable parameters.
- Saliency loss weight `0.10`; divergence margin `0.05`.
- Fixed median saliency floor during policy calibration.
- Run: `20260928T161159Z-horizon-router`, three requested epochs, epoch 1 selected.

### Enhanced profile

- 25% clean / 75% individual-corruption mixture spanning severities 1–3.
- Final three ViT blocks trainable.
- Backbone learning rate `1e-5`; classifier and saliency heads `5e-5`.
- Label smoothing `0.05`.
- Saliency loss weight `0.25`; divergence margin `0.08`.
- Joint selection of saliency-floor quantile and route threshold on 2011.
- Run: `20260929-enhanced-3pp`, five requested epochs, epoch 4 selected.

The enhanced profile was introduced to test a predeclared target of at least a 3.00 percentage-point improvement over baseline ViT on 2014 combined drift.

## Completed results

### Standard profile: macro F1

| Condition/year | Baseline | Historical XAI | Updated XAI | Routed |
|---|---:|---:|---:|---:|
| Clean 2012 | 95.86 | 95.96 | 96.56 | 96.40 |
| Clean 2013 | 95.27 | 95.23 | 96.13 | 95.79 |
| Clean 2014 | 95.13 | 95.24 | 95.85 | 95.46 |
| Combined 2012 | 95.47 | 95.47 | 96.17 | 96.08 |
| Combined 2013 | 93.75 | 93.81 | 94.71 | 94.55 |
| Combined 2014 | 90.59 | 90.34 | 91.75 | 91.50 |

On 2014 combined drift, the updated model gains 1.41 points over historical XAI. Routing retains 1.16 points, or 82.3% of the available update gain.

### Enhanced profile: macro F1

| Condition/year | Historical XAI | Enhanced XAI | Enhanced routed | Helpful/harmful routes |
|---|---:|---:|---:|---:|
| Clean 2012 | 95.96 | 95.89 | 96.03 | 65 / 62 |
| Clean 2013 | 95.23 | 95.49 | 95.55 | 60 / 44 |
| Clean 2014 | 95.24 | 95.47 | 95.30 | 69 / 64 |
| Combined 2012 | 95.47 | 95.71 | 95.81 | 76 / 58 |
| Combined 2013 | 93.81 | 94.43 | 94.44 | 101 / 66 |
| Combined 2014 | 90.34 | 91.99 | 91.85 | 188 / 90 |

2014 combined-drift conclusions:

- Enhanced XAI versus original baseline ViT: **+1.40 points**.
- Enhanced XAI versus historical XAI: **+1.65 points**.
- Enhanced routed versus original baseline ViT: **+1.26 points**.
- Router retained **91.5%** of the enhanced update gain over historical XAI.
- H1, the 3-point target: **not met**.
- H2, at least 75% gain retention: **met**.
- H3, no clean decline greater than 0.50 points: **met**.

Enhanced checkpoint selection and timing:

- Epoch 1 calibration macro F1: 93.96%; 150.60 s.
- Epoch 2: 94.26%; 138.16 s.
- Epoch 3: 94.04%; 138.25 s.
- Epoch 4: **94.39%**, selected; 138.05 s.
- Epoch 5: 93.00%; 137.82 s.

The drop at epoch 5 indicates overfitting after the selected checkpoint. The enhanced profile adds only 0.24 points to the always-new 2014 combined score and 0.35 points to routing compared with the standard profile.

## Reporting artifacts

- Markdown paper: `reports/transparent_router/long_horizon_mock_paper.md`
- PDF paper: `output/pdf/transparent_router_mock_paper.pdf`
- Compact results: `reports/transparent_router/long_horizon_summary.json`
- Standard policy: `reports/transparent_router/router_policy.json`
- Standard saliency panels: `reports/transparent_router/20260928T161159Z-horizon-router/`
- PDF/Markdown generator: `scripts/build_mock_paper.py`

Commit history milestones:

- `9fa9159`: fixed three-year drift horizon.
- `804123f`: report and standard saliency artifacts.
- `42ca356`: enhanced adaptation profile.
- `b2ad2f5`: expanded seven-page research outline.
- `b998f7b`: enhanced results added to the paper and summary.

## Server state observed on 2026-10-05

- Server checkout is clean at `b2ad2f5`; it is one report-only commit behind local/GitHub `b998f7b`.
- No tmux sessions are active.
- `/mnt/sdd` has approximately 212 GiB free and is 9% used.
- Fork runs occupy approximately 1.3 GiB.
- Enhanced artifacts include two ~88.6 MB checkpoints, a ~21.2 MB per-sample drift JSON, a ~1.78 MB future-test JSON, and a ~1.70 MB routing CSV.
- Tqdm-heavy pipeline logs range from roughly 0.9 MB to 4.8 MB because carriage-return progress updates are captured by `tee`.
- The enhanced server run has not yet had its own bounded saliency panels exported into the Git report folder.

## Highest-priority scientific improvements

1. **Make the profile comparison budget-matched.** The standard run requested three epochs and the enhanced run five. Re-run both with the same maximum epoch budget and identical early-stopping/checkpoint-selection rules. The current enhanced comparison is informative but not a clean controlled ablation.
2. **Separate checkpoint selection from router calibration.** Implemented on 2026-10-09: future runs now use deterministic, class-stratified, disjoint 2011 partitions by default. `--reuse-calibration-for-selection` is retained only for reproducing archived exploratory runs.
3. **Run multiple seeds and uncertainty estimates.** At least three seeds, paired per-sample bootstrap confidence intervals, and year-level effect sizes are needed before claiming improvement.
4. **Use a factorial ablation.** Independently toggle corruption curriculum, number of trainable blocks, saliency loss, and adaptive routing. The enhanced profile changes all four, preventing causal attribution.
5. **Introduce a stronger saliency-specific intervention.** The capacity increase barely changed results. The next predeclared method should test saliency-guided patch corruption/masking or clean-versus-drift saliency consistency, with a matched non-saliency masking control.
6. **Measure explanation faithfulness directly.** CLEAR-100 lacks object masks. Add deletion/insertion curves, sufficiency/comprehensiveness, or evaluate localization on a secondary dataset with boxes/masks. Do not call top-20% map overlap object IoU.
7. **Separate real temporal drift from synthetic stress tests.** Report clean year-to-year degradation as the real temporal benchmark and synthetic corruption as a controlled robustness layer. Quantify shift severity rather than relying only on year labels.
8. **Validate on a second temporal image dataset.** This is needed to assess whether the learned update rule transfers beyond CLEAR-100.

## Highest-priority engineering improvements

1. Add a server sync helper based on `git bundle` or configure a read-only deploy key; current HTTPS pulls fail without GitHub credentials.
2. Export enhanced-run saliency panels and a compact enhanced policy/metric summary. Avoid committing the 21 MB per-sample JSON or 88 MB checkpoints.
3. Prevent noisy progress-bar control characters in persistent logs. Keep tqdm for the attached terminal while writing structured JSONL events or a line-oriented non-interactive log.
4. Add a comparison command that validates equal seed, epoch budget, batch size, temporal splits, and checkpoint rule before producing a standard-versus-enhanced table.
5. Add deterministic stratified splitting for the 2011 checkpoint-selection and router-calibration subsets.

## 2026-10-09 literature and protocol revision

- The supplied Word draft was expanded with numbered introduction subsections, a standalone literature review, quantitative prior-effect comparisons, and a design-decision matrix.
- The Siamese claim is now bounded precisely: this is an asymmetric Siamese evaluation architecture with two same-family ViTs and an explicit logistic comparison layer, not a hidden end-to-end metric network.
- Prior results remain labelled exploratory because adapted-checkpoint selection and router fitting shared the 2011 labels.
- `fyp-train-router` now defaults to deterministic, class-stratified, disjoint selection and router-calibration partitions. A legacy flag exists only to reproduce the archived runs, and resume operations reject protocol mismatches.
- The completed source-separated results are interpreted against prior reported effect sizes rather than against a universal three-point threshold. The document now explains that task and metric differences prevent direct cross-paper ranking.
- Final editable review artifact: `output/docx/updated_ViT_router_mock_paper_2_literature_revision.docx`.
6. Record Git commit SHA and full method hyperparameters in every run config, including default profile names rather than `null`.
7. Add automatic compact summaries and bounded report export at run completion.

## Recommended next execution order

1. Fix the 2011 data reuse and budget mismatch.
2. Add saliency-guided masking plus a matched random-masking control.
3. Run a small code-path validation.
4. Run standard, random-mask control, and saliency-mask method with the same three seeds and epoch budget.
5. Freeze outputs, compute paired confidence intervals, and export enhanced saliency panels.
6. Update the paper only after the preregistered comparison is complete.

## 2026-10-05 checkpoint and controlled-ablation follow-up

Implementation additions:

- `RunStore.save_training_checkpoint` atomically writes `models/resume.pt` after every evaluated epoch and records `resume.json` metadata. The checkpoint contains the newer ViT, AdamW optimizer, completed epoch, and best score.
- `fyp-train-router --resume-run-dir` restores the next epoch from that state. A deliberate stop after standard epoch 2 resumed at epoch 3 and later selected epoch 4.
- `--disable-saliency-loss` sets the pairwise divergence and complementary-evidence auxiliary weights to zero. It does not remove the learned saliency gate from the classifier forward pass.
- `--drift-scenarios clean combined` supports focused final ablations.
- Per-sample drift routing decisions are now written to `logs/drift_routing_decisions.csv`.
- `fyp-analyze-saliency` emits outcome-stratified CSV/JSON, a compact headline JSON, and a box-plot PNG.
- `fyp-compare-runs` reports scores, policies, epoch traces, and explicit equality checks for the fixed protocol controls.
- `tests/test_run_store.py` verifies model weights, optimizer state, epoch, score, and metadata survive a checkpoint round trip.

All three comparison runs match on five epochs, batch size 8, seed 42, adaptation year 2010, calibration year 2011, test years 2012–2014, and long-horizon mode. All selected epoch 4.

| Five-epoch configuration | Best 2011 F1 | 2014 clean routed | 2014 combined new | 2014 combined routed | Helpful / harmful | Routed to new |
|---|---:|---:|---:|---:|---:|---:|
| Matched standard | 95.04 | 95.78 | 91.68 | 91.72 | 144 / 57 | 299 |
| Enhanced with auxiliary saliency losses | 94.39 | 95.30 | 91.99 | 91.85 | 188 / 90 | 388 |
| Enhanced without auxiliary saliency losses | 94.63 | 95.45 | 91.86 | 91.62 | 125 / 43 | 245 |

Interpretation:

- Equalizing the epoch budget reduces the enhanced-versus-standard difference to **+0.31 points always-new** and **+0.13 points routed** on 2014 combined drift.
- Holding the enhanced architecture and curriculum fixed, the explicit auxiliary saliency objectives contribute **+0.13 points always-new** and **+0.23 points routed** versus the no-auxiliary-loss control.
- The full enhanced router trades higher coverage and more helpful corrections for more harmful switches. The no-auxiliary control has a higher helpful/harmful ratio (2.91 versus 2.09) but a lower overall routed macro F1.
- In the no-auxiliary control, helpful routes have mean saliency JS 0.0607, harmful routes 0.0512, and retained historical cases 0.0473. Mean route probabilities are 0.821, 0.767, and 0.310. Distributions overlap, so saliency change is a clear routing eligibility signal but only a modest discriminator of useful versus harmful updates.
- Inspected full-enhanced panels show interpretable but often contextual evidence: bowling lanes, cloud structure, or framed display regions may receive more weight than a single object. These learned patch gates are not segmentation masks.

New Git-friendly artifacts are under `reports/transparent_router/20261005-ablation/`: compact comparison JSON, two saliency-analysis bundles, and 90 bounded full-enhanced/control panels. Large checkpoints and raw future JSON remain server-side.

## 2026-10-05 counterexample audit

`fyp-audit-router` now evaluates the complete 4,999-image 2014 combined-drift stream while retaining manifest indices. The enhanced run contains 188 helpful routes, 90 harmful routes, 17 missed helpful switches, and 9 cases where retaining the historical model prevents a newer-model error. The router therefore makes 98 net corrections.

For all 304 cases where one model is correct and the other is wrong, an equal-area deletion test removes either the top or bottom 20% of patches under each model's own saliency map. The new-model mean faithfulness margin is 0.175 on helpful routes and 0.010 on harmful routes; its positive-margin fractions are 72.3% and 57.8%. The historical-model mean margin reverses from 0.048 on helpful routes to 0.182 on harmful routes. This supports an outcome-level association between correct predictions and behaviourally consequential saliency, but it does not establish object localization or causal attribution.

Artifacts are in `reports/transparent_router/20261005-counterexamples/`: `all_decisions.csv`, `selected_cases.csv`, `counterexample_audit.json`, and 24 bounded six-panel images. The editable Word report incorporates the supplied saliency-guided attention, XGD, Siamese-ViT, explanation-guided distillation, saliency-guided warping, saliency-guided training, and asymmetric Siamese papers as design rationale.

## Drift protocol v2

The original `combined` condition visibly overemphasized occlusion because every sample received a square local-mean mask. It is retained unchanged only for reproduction. The v2 benchmark separates illumination, white balance, signal-dependent sensor noise, defocus blur, motion blur, JPEG compression, resolution loss, viewpoint shift, and occlusion. Its `realistic_mix` applies one, two, or three non-occlusion sources at severities one, two, or three and adds occlusion to a deterministic 10% minority of samples. Every mixed sample logs its component names.

The `realistic` adaptation profile uses 25% clean samples and 75% samples with one individual source. Occlusion occupies 1/12 of the full adaptation stream. Compound mixtures are held out until the 2012–2014 evaluation. The legacy transform, new sources, rationale, and limitations are documented in `docs/drift_protocol_v2.md`.

Server preflight on the configured uv environment passed four unit tests covering checkpoint recovery, output shape and finiteness, deterministic transforms, varied mixture compositions, and the declared minority occlusion rate. The source-labelled severity-three preview is under `reports/transparent_router/drift-v2-preflight/` and shows a mixture of JPEG compression, motion blur, and sensor noise without occlusion for the inspected sample.

## 2026-10-06 drift-v2 matched result and document revision

The resumed saliency-supervised and no-auxiliary runs completed the same source-separated 2012–2014 evaluation from one frozen historical checkpoint. Equality of the historical scores was checked before comparison. On the 2014 horizon, the saliency-supervised routed model changes macro F1 by +3.45 points under motion blur, +2.03 under `realistic_mix`, +0.75 under occlusion, +0.59 under viewpoint shift, and +0.12 on clean images. The no-auxiliary routed control gains +2.55 under motion blur and +1.19 under `realistic_mix`. Auxiliary saliency objectives therefore add +0.90 and +0.84 routed points in those two difficult conditions, while their clean routed effect is −0.18 points. The always-new motion-blur models are nearly tied (+3.96 versus +3.94), locating the main auxiliary difference in conditional routing rather than aggregate updated-classifier accuracy.

Full 2014 deletion audits were completed for motion blur, viewpoint shift, and `realistic_mix`. Their helpful/harmful/missed counts are 357/90/47, 88/56/17, and 261/95/30. Updated-map mean deletion margins on helpful versus harmful routes are 0.147/0.031, 0.190/−0.053, and 0.188/0.014. Historical-map margins reverse toward harmful routes in all three sources. These group-level reversals support paired explanation behaviour as routing evidence, but the overlap and remaining harmful routes prohibit a causal or object-localization claim.

Compact results, selected panels, the matched forest plot, and the multi-source helpful-case figure are stored under `reports/transparent_router/20261006-drift-v2/`. Raw future-evaluation JSON and full per-sample decision tables remain outside Git. The editable paper now contains separate technical diagrams for the embedded XAI ViT forward path and the asymmetric historical/update Siamese router, plus source-specific results and faithfulness analysis.

## 2026-10-06 supplied-paper router plot revision

The user-supplied `updated_ViT_router_mock_paper_2.docx` is the canonical formatting base for this revision. Its 23-page layout and existing figures were preserved. Only Figure 3, its caption, and the adjacent interpretation paragraph were updated; all other embedded media retained their hashes and dimensions. The revised editable document is `output/docx/updated_ViT_router_mock_paper_2_router_plot.docx`.

The replacement figure separates two questions. The first panel shows the routed macro F1 gain over the frozen historical ViT, which is the intended deployment baseline. The second isolates the effect of auxiliary saliency supervision by comparing the saliency router with the matched no-auxiliary control. The evidence supports a qualified conclusion: routed inference beats the historical ViT in all 11 conditions, with gains of +3.45 points under motion blur and +2.03 points under the realistic mixture. It retains 87.1% and 91.9% of the unconditional updated model's gains in those two conditions, but trails always-new inference by 0.06–0.51 points across the full table. Auxiliary saliency losses add +0.90 routed points for motion blur and +0.84 for the realistic mixture, while clean and mild-shift effects are small or negative.

The paper therefore presents routing as a conservative and auditable deployment policy rather than the universal accuracy winner. Helpful/harmful/missed switch counts remain visible in the prose so the selective-update cost is explicit. The concise GitHub-facing record is `reports/transparent_router/20261006-drift-v2/router_tradeoff_summary.md`.

## 2026-10-06 saliency visualisation correction

The earlier Figure 4 overlays used a 0.28 single-colour blend and made the historical and updated evidence maps almost indistinguishable on the printed page. The audit renderer now applies robust within-map 5th–95th percentile normalisation, an inferno-style 0.58 image blend, and a separate signed difference map. Red in the difference panel denotes increased updated-model patch weight; blue denotes reduced weight. The display scaling reveals spatial redistribution without changing predictions, routes, or metrics. Absolute change remains represented by Jensen–Shannon divergence, and behavioural relevance by the top-versus-bottom deletion margin.

All three 2014 source audits were regenerated from the saved checkpoint on the server. The selected sample identities and aggregate counts remain unchanged. Nineteen bounded PNG artifacts—the six selected panels for each source plus the three-source composite—were refreshed for Git review; full decision tables remain server-side.

## 2026-10-11 package refactor and thesis-format revision

The prototype now uses an installable `src/fyp_drift` package with separate data, model, training, evaluation, artefact, visualisation, workflow, and system modules. User-facing operations are exposed through `fyp-*` console commands in `pyproject.toml`; deployment scripts invoke those commands through uv. The final local and server checks passed all six tests. Local, GitHub, and `/home/ubuntu/FYP-fork` were synchronised at commit `6f2cd1b` before the document-only revision, and no training or tmux process was started.

The editable thesis draft `output/docx/updated_ViT_router_mock_paper_2_literature_revision_uk.docx` uses UK spelling in the authored prose, Times New Roman 12 point throughout, 1.5 line spacing, and justified body text. The revision adds explicit rationales for DINOv2-small, retaining historical and adapted branches, and every enhanced-adaptation setting. It preserves all eight embedded figures byte for byte. A 30-page LibreOffice render was checked page by page; the architecture diagrams, result plot, saliency panels, tables, references, and pagination are intact.
