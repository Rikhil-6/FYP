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
2. **Separate checkpoint selection from router calibration.** Both currently use 2011 labels. Split 2011 deterministically by class or use nested calibration so checkpoint choice and route-threshold tuning do not reuse the same examples.
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
6. Record Git commit SHA and full method hyperparameters in every run config, including default profile names rather than `null`.
7. Add automatic compact summaries and bounded report export at run completion.

## Recommended next execution order

1. Fix the 2011 data reuse and budget mismatch.
2. Add saliency-guided masking plus a matched random-masking control.
3. Run a small code-path validation.
4. Run standard, random-mask control, and saliency-mask method with the same three seeds and epoch budget.
5. Freeze outputs, compute paired confidence intervals, and export enhanced saliency panels.
6. Update the paper only after the preregistered comparison is complete.
