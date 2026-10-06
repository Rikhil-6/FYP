# Using Saliency to Route Between Historical and Updated Vision Transformers

*A fixed-horizon study of image classification under temporal drift*

## Abstract

This project asks when an image classifier should be updated after its input distribution changes. A frozen historical Vision Transformer (ViT) is compared with a second ViT adapted on later data. Both models expose the patch weights used in their predictions, and a logistic router uses changes in saliency, confidence, entropy, and class evidence to decide which prediction to keep. The models are trained and calibrated on data up to 2011, then evaluated without further tuning on 2012–2014. On the 2014 test set, routing improves macro F1 over the historical model by 3.45 percentage points under motion blur and by 2.03 points under a realistic mixture of acquisition changes. The change on clean images is 0.12 points. A matched ablation shows that the auxiliary saliency losses account for 0.90 routed points under motion blur and 0.84 points under mixed drift, although they do not help every condition. The original three-point target was not met on the earlier combined test, so the motion-blur result is reported separately. Overall, the method is useful when drift causes a clear failure in the historical model, but the routing evidence remains imperfect and still produces harmful switches.


## 1. Introduction

Image classifiers are usually trained as if future images will resemble the training set. In practice, cameras are replaced, compression settings change, lighting and weather vary, and objects are photographed from new viewpoints. ImageNet-C shows how common noise, blur, weather, and digital degradation can reduce classification accuracy [1]. CLEAR records a different problem: the visual appearance of classes changes naturally over a decade [2]. WILDS reports related shifts across locations, institutions, and time [3]. These changes can leave a deployed classifier working well on average while failing on particular classes or acquisition conditions.

Updating the model is an obvious response, but an update can fix some images and break others. This work therefore keeps the historical model as a frozen reference. A second model is adapted on a later time window, and the two predictions are compared for each image. The final decision is made by a small logistic router whose inputs can be inspected directly. This separates three questions that are often blurred together: whether the new model is better overall, whether a particular prediction should be replaced, and whether the model's explanation changed in a meaningful way.

The saliency map is part of the model rather than a picture added after inference. Yao and Gong use saliency priors to limit attention spreading into irrelevant regions [4]. XGD combines prediction alignment with saliency-based feature refinement and tests those losses through ablation [5]. Work on explanation-guided distillation likewise treats explanations as information that can be transferred between models [6]. Ismail et al. connect saliency to behaviour by perturbing low-saliency features while preserving the prediction [7], while Saliency Guided Image Warping shows how domain adaptation can become dominated by background change [8]. These studies informed two design choices here: patch weights contribute directly to the class prediction, and their usefulness is checked through patch deletion rather than visual appearance alone.

The paired comparison follows the broad idea of a Siamese network. Siamese-ViT uses transformer branches to compare changed views [9], while asymmetric Siamese networks give the two observations different roles [10]. The same distinction is useful for temporal drift. Both branches use the same ViT architecture and start from the same checkpoint, but only the newer branch is adapted. The historical branch records established behaviour; the newer branch is a candidate replacement. No convolutional branch is used.

This project tests the claim that a drift-adapted ViT can be introduced more cautiously when it is compared with a frozen historical ViT before each prediction is replaced. Saliency is treated as one routing signal, alongside confidence and predictive uncertainty. It is not assumed to prove that the updated model is correct.

This thesis is examined through three research questions:

1. **RQ1:** Does temporal adaptation improve future-year macro F1 and macro IoU under natural and controlled image drift without materially degrading clean performance?
2. **RQ2:** Can a transparent router retain most of the newer model's improvement while avoiding cases where the historical model was already correct?
3. **RQ3:** Do the explicit saliency objectives add measurable value beyond the same architecture, data curriculum, trainable blocks, and epoch budget without those objectives?

The experiments use only ViT models, preserve the temporal order of the data, log every routing input, and report helpful, harmful, and missed switches. The aim is to test a traceable update procedure rather than claim state-of-the-art classification accuracy.


## 2. System design

The prototype contains three ViT-only variants:

1. **Baseline ViT:** a frozen historical classifier.
2. **XAI ViT:** the same class of model with attention-derived saliency used during learning and evaluation.
3. **Transparent update router:** a historical XAI ViT and a drift-adapted XAI ViT evaluated as a pair. A calibrated logistic rule selects the newer prediction only when its visible evidence passes the learned threshold and a minimum saliency-change floor.

All three variants use the same DINOv2-small transformer backbone. This controls architecture as a source of variation: the comparison changes the patch aggregation, temporal adaptation, and routing policy while keeping the underlying feature family fixed. The baseline is therefore a conventional ViT reference, not a CNN reference.

The router uses four signals. **Saliency Jensen–Shannon divergence** measures the difference between the historical and updated patch-weight distributions [11]. It is near zero when the models weight the same patches and increases as their spatial evidence separates. **Confidence gain** is the change in maximum softmax probability. **Entropy reduction** is `H(p_old) − H(p_new)`, where `H(p) = −Σ p(c) log p(c)` [12]; a positive value means the updated prediction is more concentrated. **Class-evidence gain** compares support for the predicted class in the selected patches with support in their low-saliency complement [5, 7]. The raw values, route probability, chosen model, final class, correctness, and maps are saved for every image.

The baseline and historical XAI models are trained once on the historical period and then frozen. They are not retrained separately for every evaluation year. This ensures that the 2012, 2013, and 2014 scores measure increasing distance from the same deployed reference. The newer branch is also fixed after 2011 calibration, so the final horizon does not become another optimization loop.

The three classifier views serve different analytical roles. Baseline ViT measures a conventional transformer pipeline without the saliency-gated head. Historical XAI controls for the embedded explanation architecture. Updated XAI measures the maximum benefit available from adaptation when it is always selected. Routed XAI then measures how much of that benefit survives a conditional, inspectable decision policy.


### 2.1 ViT classifier and embedded explanation

DINOv2-small follows the Vision Transformer representation introduced by Dosovitskiy et al. [13] and the self-supervised DINOv2 pretraining procedure [14]. The local data pipeline first converts every image to RGB, resizes it deterministically to 224 × 224 pixels, converts channel values to the zero-to-one range, and normalizes them with ImageNet channel means `(0.485, 0.456, 0.406)` and standard deviations `(0.229, 0.224, 0.225)`. A direct square resize was chosen to keep the prototype deterministic and inexpensive, although its possible aspect-ratio distortion is a limitation that should be tested against resize-and-crop preprocessing in later work.

The prepared image is divided into non-overlapping 14 × 14 pixel patches, producing a 16 × 16 grid, or 256 patch tokens. Each patch is projected into a 384-dimensional embedding. Positional information preserves where the patch came from, and repeated self-attention blocks allow every token to exchange information with every other token. This global token interaction is the main architectural difference from a CNN, which builds representations through local convolutional filters and progressively expanding receptive fields. A learned class token travels through the same transformer blocks and acts as a global image summary, while the 256 patch tokens retain a spatial index that can be reshaped back to the 16 × 16 grid.

The baseline classifier gives every patch equal weight. It averages the 256 patch vectors, concatenates that result with the global class token, and applies a linear 100-class head. The XAI variant replaces the uniform average with a learned gate. For patch representation *hᵢ*, a LayerNorm, linear layer, and sigmoid produce a non-negative gate *gᵢ*. Dividing each gate by the sum of all gates produces the normalized saliency weight *sᵢ*. The selected representation *v* is the weighted sum of all patch representations. In compact form, *sᵢ = gᵢ / Σⱼgⱼ* and *v = Σᵢsᵢhᵢ*. The classifier receives the class token and *v* together. Consequently, changing a patch weight changes the feature vector used to calculate the class logits. The displayed 16 × 16 heatmap is a resized view of these same normalized weights, rather than a separate visualization computed after prediction.

Figure 1 shows the complete forward path. DINOv2 returns one 384-dimensional class token and 256 patch tokens. The gate assigns one scalar to each patch, preserving the 16 × 16 spatial grid. These values are normalised to sum to one before the patch vectors are combined. The classifier receives the class token and the weighted patch sum. It is also run on a complementary sum made from *1 − sᵢ*, using the same head so that both outputs are comparable in the same class-logit space.

![Figure 1. Embedded saliency-guided ViT forward computation and training signals](xai_vit_forward.png)

For faithfulness training, the model also classifies the complementary low-saliency patches. The loss rewards greater evidence for the true class in selected patches than in their complement. During update training, a pairwise term keeps the new and historical maps close when the historical prediction is correct and requires a minimum divergence when it is wrong.

The learned map answers a limited but concrete question: which ViT patch tokens received weight in the representation used for classification? It is not treated as a segmentation mask. A concentrated map can still be wrong, and a diffuse map can still support a correct prediction. The report therefore avoids interpreting visual neatness as proof of faithfulness.

The complementary path provides a simple testable expectation: the selected patches should support the correct class more strongly than the remaining patches. This is added as a soft loss so that useful contextual information is not excluded by a hard mask.

During adaptation, the frozen branch provides the reference map. If its prediction is correct, the new map is discouraged from moving unnecessarily. If it is wrong, the pairwise loss allows a larger change. The class label still determines whether an update succeeds; map divergence alone does not.


### 2.2 Asymmetric Siamese comparison and explicit routing rule

For each image, the old and new ViTs return a class distribution and a patch-weight distribution. Four differences are standardised and passed to a saved logistic equation. The new prediction can be selected only when the models disagree, saliency divergence exceeds a calibrated floor, and the logistic probability exceeds its threshold. The coefficients and thresholds are stored in `router_policy.json`.

Figure 2 uses two instances of the same XAI-guided ViT on the same image. The setup is asymmetric. Both branches begin from the historical checkpoint, but the old branch is frozen while the new branch adapts its final three transformer blocks, saliency gate, and classification head on the 2010 stream. During evaluation, both branches are fixed.

The models are compared at the prediction and patch levels. Prediction features describe class disagreement, confidence gain, and entropy reduction. Patch features describe Jensen–Shannon divergence and the change in selected-versus-complement class evidence. No learned routing embedding is used. The saved row for each image is enough to reconstruct the routing decision.

![Figure 2. Asymmetric Siamese comparison between the frozen historical ViT and drift-adapted ViT](architecture_overview.png)

Logistic regression was chosen so that the route can be reproduced from a short coefficient vector [15]. A positive calibration target means that the historical model is wrong and the updated model is correct. The four inputs are standardised, and class-balanced fitting compensates for the smaller number of useful updates. The score is *p(update | x) = sigmoid(b + wᵀz)*. Saving *b*, *w*, and the feature means and scales is sufficient to recalculate it.

Probability alone does not determine the route. The implementation searches saliency-floor quantiles and probability thresholds on 2011 only, choosing the pair with the highest calibration macro F1 and then fewer harmful routes as the tie-breaker. At inference, the newer branch is selected only if the branches disagree, saliency change exceeds the stored floor, and the fitted probability exceeds the stored threshold. This staged rule reflects the calibration literature's warning that a classifier score should not automatically be treated as a trustworthy probability without empirical checking [16]. It also keeps the final decision auditable: a reviewer can see the raw signals, standardized values, weighted sum, two eligibility checks, and selected branch for every image.

Saliency divergence measures how much the spatial evidence distribution changed. Confidence gain measures whether the newer branch is more certain in its selected class. Entropy reduction measures whether its full predictive distribution became more concentrated. Class-evidence gain measures whether the newer representation provides stronger support for the candidate class. No feature is assumed to be sufficient in isolation.

Coefficient signs describe associations in the 2011 calibration sample. A positive confidence coefficient means that confidence gain was associated with useful updates after the other features were considered. It does not prove that increasing confidence would cause correctness. The router is transparent as a policy, while its statistical relationships remain empirical.

## 3. Temporal protocol

| Stage | Years | Role |
|---|---:|---|
| Historical training | 2004–2008 | Fit the original ViT models |
| Model selection | 2009 | Choose checkpoints without touching future years |
| Drift adaptation | 2010 | Update the newer XAI ViT with a balanced clean/corrupted stream |
| Router calibration | 2011 | Fit the explicit routing rule and threshold |
| Frozen evaluation | 2012–2014 | Test clean and drifted data once across a future horizon |

The original comparison uses a frozen `combined` condition: modest saturation change, Gaussian sensor noise, and partial occlusion. Individual corruptions appear during adaptation and calibration, while their composition is withheld until final testing. The later source-separated evaluation keeps the same chronology and checkpoint but reports each acquisition change separately.

The clean stream and the corrupted stream answer different questions. Clean year-by-year evaluation measures performance on the naturally ordered dataset. Corruption tests measure sensitivity when plausible visual degradation is added to that chronology. Synthetic corruption is therefore not presented as the source of the underlying temporal drift.

Acquisition drift arises when the image-forming process changes while the class meaning remains stable. Replacing a camera can alter sensor noise, lens blur, colour response, resolution, and compression; moving that camera can change viewpoint and background; weather, season, and lighting alter illumination and white balance. These are covariate changes because the observed pixels change even when the depicted object retains the same label. ImageNet-C formalizes several sensor, optical, environmental, and digital failure modes [1]. WILDS records naturally occurring changes across camera locations, institutions, geography, and time [3], while camera-trap studies show that a classifier can rely on location-specific backgrounds and fail at unseen camera sites [18]. CLEAR supplies the separate chronological component by preserving real image years [2].

Within the original stress test, saturation represents illumination or camera colour processing, Gaussian noise approximates sensor and transmission degradation, and partial occlusion represents obstruction, cropping, dirt, or an intervening object. Severity is bounded and logged so that the stress test remains interpretable rather than making the original image unrecognizable. These perturbations are controlled proxies for acquisition mechanisms, not proof that a specific physical camera was replaced. Occlusion is only one possible acquisition shift and should not dominate the scientific claim.

The source-separated protocol covers illumination, white-balance change, sensor noise, defocus blur, motion blur, JPEG compression, resolution loss, mild viewpoint shift, and localized occlusion. A separate realistic mixture combines a small number of these changes at bounded severity, with occlusion limited to a minority of images. This makes optical, sensor, digital, geometric, and obstruction-related shifts visible in separate rows instead of allowing one dramatic corruption to represent the whole problem. The protocol was run only after the checkpoints and routing rule had been fixed; its results appear in Section 4.3. Natural distribution-shift benchmarks also show that conclusions can differ across acquisition sources [19].

The combined condition is reserved for final evaluation. Adaptation and calibration use individual corruption families, so success on the composition measures limited generalization beyond the exact perturbation pattern seen during updating.


### 3.1 Enhanced correction experiment

The newly added `--adaptation-profile enhanced` mode is a predeclared attempt to increase the separation between the historical ViT and the saliency-responsive correction model without changing the test set. It changes only the 2010 adaptation procedure:

- 25% clean and 75% individually corrupted samples, spanning severities 1–3;
- the final three transformer blocks are trainable instead of only the final block;
- backbone learning rate remains `1e-5`, while classifier and saliency heads use `5e-5`;
- label smoothing is `0.05` and the saliency terms receive weight `0.25`;
- the saliency divergence margin increases from `0.05` to `0.08`;
- router calibration jointly selects a saliency-floor quantile and probability threshold on 2011 only.

The completed enhanced run reused the same historical checkpoint, seed 42, 2011 calibration set, and frozen 2012–2014 evaluation streams. Combined corruption remained unavailable during adaptation and calibration. Epoch 4 was selected by 2011 macro F1 (**94.39%**); epoch 5 declined to **93.00%**, indicating overfitting after the selected checkpoint. Training took approximately **138 seconds per epoch** after warm-up.


### 3.2 Hypotheses and success criteria

- **H1:** enhanced saliency adaptation improves 2014 combined-drift macro F1 by at least **3.00 percentage points** over the original baseline ViT.
- **H2:** transparent routing retains at least 75% of the gain achieved by always using the updated model.
- **H3:** clean 2014 macro F1 does not fall by more than 0.50 points relative to the historical XAI model.

The 3-point value is a target fixed before viewing the enhanced results. Failure to reach it must be reported as a negative result rather than addressed by tuning on 2012–2014 labels.


### 3.3 Enhanced result against the hypotheses

- **H1 not met:** the updated model improved by **1.40 points** over baseline ViT, below the 3.00-point target.
- **H2 met:** routing retained `(91.85−90.34)/(91.99−90.34) = 91.5%` of the newer model's gain over historical XAI.
- **H3 met:** routed clean 2014 macro F1 was **95.30%**, 0.06 points above historical XAI.


### 3.4 Checkpoint and interruption recovery

Every evaluated epoch atomically replaces `models/resume.pt` and writes a small `resume.json`. The binary checkpoint contains the newer ViT parameters, optimizer state, completed epoch, and best validation score; the separate best-model file remains the checkpoint used for final evaluation. The matched standard run was deliberately stopped after epoch 2 and resumed at epoch 3. It retained the epoch-2 optimizer and best score, later selected epoch 4, and completed the frozen horizon normally. A local round-trip regression test independently reloads and compares weights, optimizer state, epoch, score, and metadata.


## 4. Main results

Macro F1 is shown as a percentage. Helpful and harmful routes count cases where choosing the newer model corrected or introduced an error, respectively.

| Scenario | Year | Baseline | Historical XAI | Updated XAI | Routed | Helpful | Harmful |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 2012 | 95.86 | 95.96 | 96.56 | 96.40 | 30 | 8 |
| Clean | 2013 | 95.27 | 95.23 | 96.13 | 95.79 | 40 | 12 |
| Clean | 2014 | 95.13 | 95.24 | 95.85 | 95.46 | 30 | 18 |
| Combined | 2012 | 95.47 | 95.47 | 96.17 | 96.08 | 45 | 14 |
| Combined | 2013 | 93.75 | 93.81 | 94.71 | 94.55 | 72 | 32 |
| Combined | 2014 | 90.59 | 90.34 | 91.75 | 91.50 | 120 | 48 |

On 2014 combined drift, the baseline reaches **90.59%**, the historical XAI model **90.34%**, the updated XAI model **91.75%**, and the transparent router **91.50%** macro F1. The router captures `(91.50−90.34)/(91.75−90.34) = 82.3%` of the available update gain. The updated model also remains stronger on clean images, although the router becomes more conservative and records 18 harmful routes against 30 helpful routes in 2014.


### 4.1 Enhanced profile results

| Scenario | Year | Historical XAI | Enhanced XAI | Enhanced routed | Helpful | Harmful |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 2012 | 95.96 | 95.89 | 96.03 | 65 | 62 |
| Clean | 2013 | 95.23 | 95.49 | 95.55 | 60 | 44 |
| Clean | 2014 | 95.24 | 95.47 | 95.30 | 69 | 64 |
| Combined | 2012 | 95.47 | 95.71 | 95.81 | 76 | 58 |
| Combined | 2013 | 93.81 | 94.43 | 94.44 | 101 | 66 |
| Combined | 2014 | 90.34 | 91.99 | 91.85 | 188 | 90 |

The enhanced curriculum improved the always-new 2014 combined score from **91.75% to 91.99%**, a further **0.24 points**. The routed score increased from **91.50% to 91.85%**, a further **0.35 points**. The modest change indicates that simply increasing adaptation capacity is insufficient to create a 3-point separation.



### 4.2 Controlled five-epoch ablation

The matched run removes the earlier three-versus-five-epoch imbalance. The auxiliary-loss control keeps the enhanced curriculum, trainable blocks, learning rates, and saliency-gated classifier fixed, but sets the pairwise saliency-divergence and complementary-evidence loss weights to zero. It therefore isolates the explicit saliency supervision while retaining saliency inside the forward pass.

| Configuration | 2014 clean routed | Combined old | Combined new | Combined routed | Helpful | Harmful |
|---|---:|---:|---:|---:|---:|---:|
| Matched standard (5 epochs) | **95.78** | **90.34** | 91.68 | 91.72 | 144 | 57 |
| Enhanced with auxiliary saliency losses | 95.30 | **90.34** | **91.99** | **91.85** | **188** | 90 |
| Enhanced without auxiliary saliency losses | 95.45 | **90.34** | 91.86 | 91.62 | 125 | **43** |

Bold type marks the best value in each evaluation column across the three matched configurations; ties are bolded together. Helpful and harmful counts describe route composition rather than interchangeable accuracy measures, so the most helpful and least harmful counts are emphasized separately.

Against the matched enhanced control, the explicit auxiliary saliency objectives changed 2014 combined macro F1 by **+0.13 points** for always-new inference and **+0.23 points** after routing. These are single-run paired differences, not confidence-bounded causal estimates.


### 4.3 Source-separated drift evaluation

The broader protocol evaluates the same saved historical checkpoint, adapted checkpoint, and 2011 routing policy across 11 independently named 2014 conditions. Each row contains the frozen historical score, the always-updated score, the transparent routed score, and the routed change from the historical branch. This separation prevents a favourable occlusion example from standing in for unrelated acquisition changes.

| 2014 condition | Historical | Updated | Routed | Routed change |
|---|---:|---:|---:|---:|
| Clean | 95.24 | 95.54 | 95.36 | +0.12 |
| Illumination | 95.13 | 95.46 | 95.24 | +0.11 |
| White balance | 95.16 | 95.61 | 95.43 | +0.27 |
| Sensor noise | 94.20 | 94.72 | 94.44 | +0.24 |
| Defocus blur | 94.41 | 94.93 | 94.75 | +0.34 |
| **Motion blur** | **83.17** | **87.13** | **86.62** | **+3.45** |
| JPEG compression | 93.64 | 94.27 | 94.09 | +0.45 |
| Resolution loss | 93.69 | 94.26 | 94.10 | +0.41 |
| Viewpoint shift | 94.32 | 95.09 | 94.91 | +0.59 |
| Occlusion | 93.93 | 94.74 | 94.68 | +0.75 |
| **Realistic mixture** | **86.41** | **88.62** | **88.44** | **+2.03** |

Motion blur produces the largest routed gain: **+3.45 percentage points**, from 357 helpful and 90 harmful switches. The realistic mixture gains **+2.03 points**, with 261 helpful and 95 harmful switches. Occlusion gains **+0.75 points**. The method is therefore not limited to occlusion, but its effect is clearly source dependent. It helps most when motion or several acquisition changes produce a sizeable failure in the historical model. Mild colour and illumination changes leave much less room for improvement.

| Matched 2014 condition | Saliency routed change | No-aux routed change | Auxiliary effect on route |
|---|---:|---:|---:|
| Clean | +0.12 | **+0.30** | −0.18 |
| Motion blur | **+3.45** | +2.55 | **+0.90** |
| Viewpoint shift | **+0.59** | +0.53 | +0.06 |
| Occlusion | **+0.75** | +0.69 | +0.06 |
| Realistic mixture | **+2.03** | +1.19 | **+0.84** |

![Figure 3. Matched 2014 change by drift source](20261006-drift-v2/drift_v2_2014_comparison.png)

The saliency and no-auxiliary runs use the same seed, historical checkpoint, temporal split, batch size, and scenario definitions. Under motion blur, their routed gains are 3.45 and 2.55 points, so the auxiliary losses add 0.90 points. Under the realistic mixture, they add 0.84 points. The always-new motion-blur models are almost tied (+3.96 and +3.94), suggesting that the larger difference appears in the routing behaviour rather than in the updated classifier alone. This effect is not consistent across all conditions: on clean data the saliency-supervised router is 0.18 points below the matched control, and several mild shifts differ by less than 0.10 points.



## 5. Saliency and routing analysis

Each audit panel shows the clean image, the drifted input, and the historical and updated evidence maps. The label line records the true class, both predictions, the routed prediction, route probability, saliency divergence, and deletion margin. The overlays are kept faint enough for the image to remain visible.

Figure 4 contains one helpful route from each of three drift families. The examples were selected for inspection and are not used to estimate performance; the aggregate counts below provide that evidence.

![Figure 4. Helpful routed corrections under motion blur, viewpoint shift, and a realistic mixture](20261006-drift-v2/multidrift_helpful_cases.png)


### 5.1 Quantitative saliency outcome analysis

For the 2014 combined-drift auxiliary-loss control, helpful routes had mean saliency Jensen-Shannon divergence **0.0607**, compared with **0.0512** for harmful routes and **0.0473** when the historical prediction was retained. Mean route probability was **0.821** for helpful routes, **0.767** for harmful routes, and **0.310** for retained historical predictions. There were **125 helpful versus 43 harmful routes** (ratio 2.91). The matched standard run showed the same within-run ordering for helpful versus harmful divergence (0.00759 versus 0.00651), although raw divergence magnitudes should not be compared across differently adapted models.

![Saliency and route probability by outcome](20261005-ablation/enhanced-nosaliency/saliency_route_outcomes.png)

The distributions overlap. Routed and retained images separate partly because the policy itself requires a minimum saliency change. That confirms the gate is being applied, but it does not show that a large map change predicts correctness. Helpful routes have somewhat larger divergence and route probability on average, yet many harmful routes fall in the same range. Saliency movement is therefore useful as a filter, but not as a decision rule by itself.

Map change measures difference, not improvement. A new model can move its attention and still be wrong, or become highly confident in the wrong class. CLEAR also provides image labels rather than object masks. Context such as a bowling lane, kitchen surface, or display frame may be useful in one image and a shortcut in another. Since both models share their pretraining and initialization, some correct updates also require only a small redistribution of patch weights.

The maps are best read as **evidence-allocation diagnostics**: they show which encoded patches received weight in the classifier. They are not object masks and do not establish that the map change caused a correction. The deletion audit adds a behavioural check by comparing the effect of removing highly weighted and weakly weighted patches. Even this test is limited because the deleted image is outside the original data distribution.


### 5.2 Counterexample and faithfulness audit

The source-specific audit applies the same top-versus-bottom 20% deletion test whenever one model is correct and the other is wrong. A positive margin means that removing the most highly weighted patches reduces true-class probability more than removing the least-weighted patches.

| 2014 source | Helpful routed | Harmful routed | Missed helpful | Net corrections | New-map margin, helpful | New-map margin, harmful |
|---|---:|---:|---:|---:|---:|---:|
| Motion blur | **357** | 90 | 47 | **267** | 0.147 | 0.031 |
| Viewpoint shift | 88 | 56 | 17 | 32 | **0.190** | −0.053 |
| Realistic mixture | 261 | 95 | 30 | 166 | **0.188** | 0.014 |

For the updated model, positive margins are more common on helpful than harmful routes: 70.0% versus 61.1% for motion blur, 72.7% versus 44.6% for viewpoint shift, and 76.6% versus 60.0% for the realistic mixture. The historical map shows the reverse pattern. Its mean margin is larger on harmful routes, where the historical prediction was correct. The reversal is consistent with the routing idea, although the overlap explains why errors remain.

The full 2014 combined-drift stream contains **4,999** images. The enhanced router makes **188 helpful switches** where the historical model is wrong and the newer model is correct, **90 harmful switches** in the opposite direction, and misses **17** available corrections. It therefore produces **98 net corrected predictions**. These counts explain why the overall score changes modestly even though some individual decisions change substantially.

For every one of the 304 samples where one model is correct and the other is wrong, the audit deletes the top 20% and bottom 20% of patches under each model's own saliency map. The faithfulness margin is the drop in true-class probability after high-saliency deletion minus the drop after low-saliency deletion. For helpful routed cases, the newer model has a mean margin of **0.175**, compared with **0.010** on harmful routed cases; **72.3%** of helpful routes have a positive new-model margin, compared with **57.8%** of harmful routes. Conversely, the historical model's mean margin is **0.048** on helpful routes and **0.182** on harmful routes. This reversal is consistent with the intuitive routing goal: the model that is correct tends to have the more behaviourally consequential map.

![Harmful switch: correct historical class 22 is replaced by class 44](20261005-counterexamples/harmful_routed/sample_11108.png)

The harmful example shows why saliency change alone is insufficient: the historical model was already correct and its map had the stronger deletion response. Patch deletion is still an artificial intervention, so this result does not prove that the highlighted pixels caused the model update.


### 5.3 Scope of the method

The intended setting is covariate drift: the label keeps the same meaning while the image-forming process or context changes. Examples include camera replacement, compression, sensor noise, white balance, illumination, viewpoint, and partial obstruction. Saliency can help only when useful class evidence remains visible and the updated model learns to weight it differently.

The source-separated results show the largest gains under motion blur and mixed drift. Viewpoint, occlusion, and digital degradation produce smaller gains, while mild colour and illumination shifts change little. The method is neither an occlusion-specific fix nor a general solution to every form of drift.

The method should not be expected to solve concept drift where the label meaning changes, new-class emergence, extreme occlusion, or systematic dataset errors. In those cases the saliency map may faithfully explain a wrong premise. A stronger next experiment should therefore add a matched random-mask control and an external temporal dataset, then compare confidence-only, saliency-only, and deletion-aware routers without tuning on the final horizon.


## 6. Discussion and limitations

The longer horizon makes the decline under difficult drift easier to see. It also shows why the historical model should not be replaced blindly: the updated model corrects many cases but still introduces errors. The routing rule can be reconstructed from the four stored inputs, coefficients, threshold, and saliency floor in `router_policy.json`. Its coefficients describe relationships in the 2011 calibration set; they are not causal explanations.

The main limitations are one random seed, one temporal dataset, synthetic stress tests, and one fitted threshold. Checkpoint selection and router calibration also reuse 2011 examples. No confidence intervals or multi-seed hypothesis tests are reported. The original combined test did not reach the three-point target; only the later motion-blur test exceeded it. Saliency losses add 0.90 routed points under motion blur and 0.84 under mixed drift, but their effect is small or negative elsewhere. These results do not support a broad causal claim about saliency.

The next experiment should repeat the temporal backtest across several seeds, add paired bootstrap confidence intervals, and reserve an external temporal dataset for final evaluation. Confidence-only, saliency-only, and deletion-aware routers should be compared against always-old and always-new baselines under the same tuning budget.


## Appendix A. Metric and routing glossary

**Macro F1** calculates F1 independently for every class and then gives each class equal weight. It is the principal classification measure because a large class cannot dominate the reported result. **Macro IoU** applies the same equal-class principle to one-versus-rest intersection over union. It remains a classification-set measure in this project and must not be confused with spatial mask IoU.

**Helpful route** means that the historical prediction was wrong, the newer prediction was correct, and the policy selected the newer branch. **Harmful route** means that the historical prediction was correct but the selected newer prediction was wrong. **Missed helpful case** means that a correct newer prediction was available but the policy retained an incorrect historical answer.

**Saliency Jensen–Shannon divergence** measures the bounded difference between the old and new normalized patch-weight distributions. **Confidence gain** compares the selected class probabilities. **Entropy reduction** measures whether the newer predictive distribution is more concentrated. **Class-evidence gain** compares the evidence produced for the candidate class.

**Retained gain** measures how much of the always-new improvement survives routing: `(routed − old) / (new − old)`. A value near one means routing preserves most of the candidate model's benefit. The ratio becomes unstable when the new and old scores are nearly equal, so the underlying scores must always be reported beside it.

These definitions are fixed across the report. They make clear which measures evaluate classification, which evaluate the decision policy, and which describe explanation change. None of the routing features is itself a ground-truth measure of drift.


## References

1. Hendrycks and Dietterich (2019), *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. https://arxiv.org/abs/1903.12261
2. Lin et al. (2021), *CLEAR: A Benchmark for Continual Learning on Real-World Imagery*. https://clear-benchmark.github.io/
3. Koh et al. (2021), *WILDS: A Benchmark of in-the-Wild Distribution Shifts*. Proceedings of the 38th International Conference on Machine Learning. https://arxiv.org/abs/2012.07421
4. Yao and Gong (2020), *Saliency Guided Self-attention Network for Weakly and Semi-supervised Semantic Segmentation*. arXiv:1910.05475.
5. Nguyen et al. (2025), *XGD: Explainable AI-Guided Knowledge Distillation with Feature Refinement for Semantic Segmentation*. 38th Canadian Conference on Artificial Intelligence.
6. Cantini, Orsino, and Talia (2024), *XAI-driven knowledge distillation of large language models for efficient deployment on low-resource devices*. Journal of Big Data, 11, 63. https://doi.org/10.1186/s40537-024-00928-3
7. Ismail, Feizi, and Corrada Bravo (2021), *Improving Deep Learning Interpretability by Saliency Guided Training*. Advances in Neural Information Processing Systems.
8. Zheng, Ghosh, and Narasimhan (2024), *Saliency Guided Image Warping for Unsupervised Domain Adaptation*. arXiv:2403.12712.
9. Cheng et al. (2026), *Siamese-ViT: A Local–Global Feature Fusion Method for Real-Time Visual Navigation of UAVs in Real-World Environments*. Remote Sensing, 18, 1556. https://doi.org/10.3390/rs18101556
10. Yang et al. (2021), *Semantic Change Detection with Asymmetric Siamese Networks*. arXiv:2010.05687.
11. Lin (1991), *Divergence Measures Based on the Shannon Entropy*. IEEE Transactions on Information Theory, 37(1), 145–151. https://doi.org/10.1109/18.61115
12. Shannon (1948), *A Mathematical Theory of Communication*. Bell System Technical Journal, 27, 379–423 and 623–656. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
13. Dosovitskiy et al. (2020), *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. https://arxiv.org/abs/2010.11929
14. Oquab et al. (2023), *DINOv2: Learning Robust Visual Features without Supervision*. https://arxiv.org/abs/2304.07193
15. Rudin (2019), *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead*. Nature Machine Intelligence, 1, 206–215. https://doi.org/10.1038/s42256-019-0048-x
16. Niculescu-Mizil and Caruana (2005), *Predicting Good Probabilities with Supervised Learning*. Proceedings of the 22nd International Conference on Machine Learning, 625–632. https://doi.org/10.1145/1102351.1102430
17. Hendrycks et al. (2019), *AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty*. https://arxiv.org/abs/1912.02781
18. Beery, Van Horn, and Perona (2018), *Recognition in Terra Incognita*. Proceedings of the European Conference on Computer Vision. https://doi.org/10.1007/978-3-030-01270-0_28
19. Taori et al. (2020), *Measuring Robustness to Natural Distribution Shifts in Image Classification*. Advances in Neural Information Processing Systems, 33.
