# Examiner review of the draft paper

## Overall assessment

The project has a credible undergraduate contribution: it links a trainable
patch weighting mechanism to prediction, compares historical and updated ViTs,
and records the evidence used by a simple routing policy. The implementation is
more substantial than a post-hoc saliency demonstration. The strongest result
is the source-separated CLEAR analysis, especially the matched control without
the auxiliary saliency losses.

The draft does not yet support a general claim that XAI improves drift
correction. Its current evidence supports a narrower conclusion: the saliency
mechanism improves selected severe CLEAR conditions, has a small positive effect
on CIFAR-10.1, does not improve Yearbook, and is neutral to slightly negative on
Camelyon17-WILDS. The paper will become stronger if it makes that claim hierarchy
explicit and treats the external screens as tests of the classifier component
rather than replications of the full router.

## Major points

### 1. Separate the three claims

The draft sometimes moves between three propositions:

1. the saliency-gated classifier can outperform uniform patch pooling;
2. temporal adaptation can outperform the historical model;
3. the transparent router can select beneficial updates.

The experiments do not test all three propositions on every dataset. CLEAR
tests the complete chain. Yearbook, CIFAR-10.1 and Camelyon17 test only matched
frozen-backbone heads and feature drift. Each result table should therefore name
the component tested and the inference it permits.

### 2. Treat the reported CLEAR run as exploratory

The published CLEAR tables reuse 2011 labels for adapted-checkpoint selection
and router calibration. The corrected code uses disjoint class-stratified
partitions, but the main numerical result has not been rerun under that protocol.
This is the largest threat to the paper's internal validity. The text should not
describe the archived thresholds as confirmatory. A final submission should
rerun at least three seeds with the corrected split and report paired confidence
intervals over image-level score differences.

### 3. Avoid post-hoc success thresholds

The two-point criterion was added after the results were available, while the
three-point target was not registered outside the project. Both values can be
useful descriptive reference points, but neither functions as a statistical
decision rule. The paper should label them as engineering targets and give
uncertainty intervals and repeated-seed effects greater evidential weight.

### 4. Prove that saliency adds information beyond confidence

The current loss ablation asks whether saliency supervision helps a
saliency-gated model. It does not answer whether saliency features improve the
router beyond confidence and entropy. The code now exposes confidence-only,
saliency-only and combined policies, but the paper has no matched result table
for them. RQ3 must remain unresolved until those policies are compared at equal
route coverage or equal harmful-route rate.

### 5. Qualify the word Siamese

The model is an asymmetric paired comparison between two versioned ViTs. It is
not a conventional Siamese metric learner trained with a contrastive or triplet
loss. Retaining the term is reasonable if the methods section states this
difference whenever the architecture is introduced. The paired design is
justified by the need to compute prediction and evidence changes against a
recoverable historical model. It should not be presented as inherently more
accurate than a single updated ViT.

### 6. Tighten drift verification

MMD, a classifier two-sample test and task performance answer different
questions. The paper handles this distinction well, but its joint detection
rule is project-specific. Report the clean null, effect size and p-value without
turning the rule into a universal definition of drift. For patch datasets,
sampling and cross-validation should respect patient or slide groups; otherwise
nearby patches can leak acquisition information across folds. The Camelyon17
screen uses patient-disjoint source partitions for this reason.

### 7. Place the external results in the results section

The detailed Yearbook and CIFAR numbers currently appear inside the protocol
section. Move them to a cross-dataset results subsection. Keep the methods
section for datasets, splits, metrics and restrictions. A compact protocol table,
a drift table and an architecture table will make the evidence easier to audit.

### 8. State the benchmark deviations

The WILDS submission rules require ten Camelyon17 replicates and prohibit
external pretraining for a standard submission. This project uses DINOv2 and
three seeds, so its Camelyon17 result is a non-standard exploratory screen. The
paper should state this before presenting the numbers. The comparison remains
useful because both heads share the same backbone, data and optimisation budget.
It should not be compared directly with the WILDS leaderboard.

### 9. Strengthen citation logic

Several references establish that a technique exists but do not establish that
it will work in this setting. The paper should distinguish three uses of a
citation:

- a benchmark paper supports the dataset and split definition;
- a methods paper supports a design choice or metric;
- the present experiments determine whether that choice helps here.

The literature can motivate saliency guidance, paired transformers and drift
tests. It cannot substitute for the missing router-feature ablation or repeated
CLEAR seeds. Claims about Camelyon17 should cite the WILDS benchmark and the
original CAMELYON17 challenge rather than a dataset mirror alone.

## Recommended submission position

The paper should present the router as an auditable model-version selection
prototype with conditional accuracy benefits. That conclusion follows from the
available evidence and leaves room for negative results. A stronger claim that
XAI provides general drift correction should wait for the corrected CLEAR runs,
the router-feature ablation and a complete external router replication.

## Sources used for this review

- WILDS benchmark and dataset definitions: <https://proceedings.mlr.press/v139/koh21a.html>
- WILDS submission rules and replicate requirements: <https://wilds.stanford.edu/submit/>
- CAMELYON17 challenge: <https://pubmed.ncbi.nlm.nih.gov/30716025/>
- Kernel two-sample testing: <https://www.jmlr.org/papers/v13/gretton12a.html>
- Classifier two-sample testing: <https://arxiv.org/abs/1610.06545>
- Interpretable-model argument: <https://doi.org/10.1038/s42256-019-0048-x>
