"""Build the compact Markdown and PDF report for the fixed-horizon experiment."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "transparent_router"
RUN = REPORTS / "20260928T161159Z-horizon-router"
PDF_DIR = ROOT / "output" / "pdf"
PAPER_MD = REPORTS / "long_horizon_mock_paper.md"
PAPER_PDF = PDF_DIR / "transparent_router_mock_paper.pdf"
SUMMARY_JSON = REPORTS / "long_horizon_summary.json"
ABLATION_DIR = REPORTS / "20261005-ablation"
ABLATION_JSON = ABLATION_DIR / "comparison.json"
SALIENCY_ANALYSIS_PNG = ABLATION_DIR / "enhanced-nosaliency" / "saliency_route_outcomes.png"

YEARS = [2012, 2013, 2014]
RESULTS = {
    "clean": {
        "baseline": [0.9586, 0.9527, 0.9513],
        "historical_xai": [0.9596, 0.9523, 0.9524],
        "updated_xai": [0.9656, 0.9613, 0.9585],
        "routed": [0.9640, 0.9579, 0.9546],
        "helpful_routes": [30, 40, 30],
        "harmful_routes": [8, 12, 18],
    },
    "combined": {
        "baseline": [0.9547, 0.9375, 0.9059],
        "historical_xai": [0.9547, 0.9381, 0.9034],
        "updated_xai": [0.9617, 0.9471, 0.9175],
        "routed": [0.9608, 0.9455, 0.9150],
        "helpful_routes": [45, 72, 120],
        "harmful_routes": [14, 32, 48],
    },
}

ENHANCED = {
    "clean": {
        "old": [0.9596, 0.9523, 0.9524],
        "new": [0.9589, 0.9549, 0.9547],
        "routed": [0.9603, 0.9555, 0.9530],
        "helpful_routes": [65, 60, 69],
        "harmful_routes": [62, 44, 64],
    },
    "combined": {
        "old": [0.9547, 0.9381, 0.9034],
        "new": [0.9571, 0.9443, 0.9199],
        "routed": [0.9581, 0.9444, 0.9185],
        "helpful_routes": [76, 101, 188],
        "harmful_routes": [58, 66, 90],
    },
}


def pp(value: float) -> str:
    return f"{value * 100:.2f}"


def load_ablation():
    """Load the compact server comparison after the controlled runs complete."""
    if not ABLATION_JSON.is_file():
        return None
    return json.loads(ABLATION_JSON.read_text(encoding="utf-8"))


def build_summary() -> None:
    policy = json.loads((REPORTS / "router_policy.json").read_text(encoding="utf-8"))
    ablation = load_ablation()
    payload = {
        "experiment": "fixed three-year temporal drift horizon",
        "train_years": [2004, 2005, 2006, 2007, 2008],
        "selection_year": 2009,
        "adaptation_year": 2010,
        "router_calibration_year": 2011,
        "held_out_test_years": YEARS,
        "macro_f1": RESULTS,
        "enhanced_macro_f1": ENHANCED,
        "enhanced_training": {
            "run_id": "20260929-enhanced-3pp",
            "selected_epoch": 4,
            "selection_macro_f1": 0.9439,
            "epoch_wall_seconds": [150.6013, 138.1634, 138.2531, 138.0546, 137.8166],
            "hypothesis_h1_three_point_gap": "not_met",
            "hypothesis_h2_retain_75_percent_gain": "met",
            "hypothesis_h3_clean_drop_at_most_half_point": "met",
        },
        "router_policy": policy,
        "source_runs": {
            "baseline": "20260928T152657Z-horizon",
            "historical_xai": "20260928T161159Z-horizon-xai",
            "transparent_router": "20260928T161159Z-horizon-router",
        },
    }
    if ablation:
        payload["controlled_ablation_20261005"] = ablation
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_markdown() -> None:
    ablation = load_ablation()
    rows = []
    for scenario in ("clean", "combined"):
        for i, year in enumerate(YEARS):
            r = RESULTS[scenario]
            rows.append(
                f"| {scenario.title()} | {year} | {pp(r['baseline'][i])} | "
                f"{pp(r['historical_xai'][i])} | {pp(r['updated_xai'][i])} | "
                f"{pp(r['routed'][i])} | {r['helpful_routes'][i]} | {r['harmful_routes'][i]} |"
            )
    enhanced_rows = []
    for scenario in ("clean", "combined"):
        for i, year in enumerate(YEARS):
            r = ENHANCED[scenario]
            enhanced_rows.append(
                f"| {scenario.title()} | {year} | {pp(r['old'][i])} | {pp(r['new'][i])} | "
                f"{pp(r['routed'][i])} | {r['helpful_routes'][i]} | {r['harmful_routes'][i]} |"
            )

    ablation_section = ""
    if ablation:
        runs = ablation["runs"]
        labels = (
            ("matched_standard", "Matched standard (5 epochs)"),
            ("enhanced_full", "Enhanced with auxiliary saliency losses"),
            ("enhanced_no_aux", "Enhanced without auxiliary saliency losses"),
        )
        comparison_rows = []
        for key, label in labels:
            clean = runs[key]["scenarios"]["clean"]["2014"]
            combined = runs[key]["scenarios"]["combined"]["2014"]
            comparison_rows.append(
                f"| {label} | {pp(clean['routed'])} | {pp(combined['old'])} | "
                f"{pp(combined['new'])} | {pp(combined['routed'])} | "
                f"{combined['helpful_routes']} | {combined['harmful_routes']} |"
            )
        full = runs["enhanced_full"]["scenarios"]["combined"]["2014"]
        no_aux = runs["enhanced_no_aux"]["scenarios"]["combined"]["2014"]
        new_effect = (full["new"] - no_aux["new"]) * 100
        routed_effect = (full["routed"] - no_aux["routed"]) * 100
        ablation_section = f"""
### 4.2 Controlled five-epoch ablation

The matched run removes the earlier three-versus-five-epoch imbalance. The auxiliary-loss control keeps the enhanced curriculum, trainable blocks, learning rates, and saliency-gated classifier fixed, but sets the pairwise saliency-divergence and complementary-evidence loss weights to zero. It therefore isolates the explicit saliency supervision while retaining saliency inside the forward pass.

| Configuration | 2014 clean routed | Combined old | Combined new | Combined routed | Helpful | Harmful |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

Against the matched enhanced control, the explicit auxiliary saliency objectives changed 2014 combined macro F1 by **{new_effect:+.2f} points** for always-new inference and **{routed_effect:+.2f} points** after routing. These are single-run paired differences, not confidence-bounded causal estimates.
"""

    markdown = f"""# Transparent Saliency Routing for Temporal Image Drift

*A compact mock research paper based on the fixed-horizon prototype run*

## Abstract

This prototype tests whether a fully vision-transformer pipeline can respond to temporal image drift while keeping its update decision inspectable. A historical ViT is trained on 2004–2008 images and selected on 2009. A newer XAI-regularised ViT is adapted on 2010 images with realistic corruptions. An explicit router is calibrated on 2011 and evaluated once on the untouched 2012–2014 horizon. In the completed enhanced run, the newer model reaches **91.99%** macro F1 under 2014 combined drift: **+1.40 points** over the original baseline ViT and **+1.65 points** over historical XAI. Routing reaches **91.85%** and retains **91.5%** of the available update gain. The predeclared 3-point hypothesis was not met.

## 1. Research question

Can a transparent, saliency-aware routing rule decide when a drift-adapted ViT should replace a historical ViT prediction, while preserving accuracy on clean data and improving performance as temporal drift accumulates?

## 2. System design

The prototype contains three ViT-only variants:

1. **Baseline ViT:** a frozen historical classifier.
2. **XAI ViT:** the same class of model with attention-derived saliency used during learning and evaluation.
3. **Transparent update router:** a historical XAI ViT and a drift-adapted XAI ViT evaluated as a pair. A calibrated logistic rule selects the newer prediction only when its visible evidence passes the learned threshold and a minimum saliency-change floor.

The router inputs are **saliency Jensen–Shannon divergence**, **confidence gain**, **entropy reduction**, and **class-evidence gain**. Every routed sample is logged with the feature values, route probability, selected model, correctness, and saliency images.

### 2.1 ViT classifier and embedded explanation

DINOv2-small produces one class token and a grid of patch tokens. The baseline averages patch evidence uniformly. The XAI model learns a scalar gate for every patch, normalises those gates into a saliency distribution, and uses the saliency-weighted patch sum together with the class token for classification. The same weights therefore affect the logits directly: the map is part of the forward computation rather than a detached visualisation.

For faithfulness training, the model also classifies the complementary low-saliency patches. The loss rewards greater evidence for the true class in selected patches than in their complement. During update training, a pairwise term keeps the new and historical maps close when the historical prediction is correct and requires a minimum divergence when it is wrong.

### 2.2 Explicit routing rule

For a sample `x`, the old and new ViTs produce predictions and saliency distributions. The policy standardises four observable differences and applies a saved logistic equation. The newer prediction is eligible only when the models disagree, saliency divergence exceeds a calibrated floor, and the logistic probability exceeds its calibrated threshold. All coefficients, means, scales, and thresholds are stored in `router_policy.json`.

## 3. Temporal protocol

| Stage | Years | Role |
|---|---:|---|
| Historical training | 2004–2008 | Fit the original ViT models |
| Model selection | 2009 | Choose checkpoints without touching future years |
| Drift adaptation | 2010 | Update the newer XAI ViT with a balanced clean/corrupted stream |
| Router calibration | 2011 | Fit the explicit routing rule and threshold |
| Frozen evaluation | 2012–2014 | Test clean and drifted data once across a future horizon |

The optional drift generator applies modest saturation shifts, Gaussian sensor noise, and partial occlusion. Individual corruptions appear during adaptation and calibration; their combined composition is withheld until final testing. This follows the corruption-testing motivation of ImageNet-C and AugMix while preserving chronological separation.

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

## 4. Main results

Macro F1 is shown as a percentage. Helpful and harmful routes count cases where choosing the newer model corrected or introduced an error, respectively.

| Scenario | Year | Baseline | Historical XAI | Updated XAI | Routed | Helpful | Harmful |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

On 2014 combined drift, the baseline reaches **90.59%**, the historical XAI model **90.34%**, the updated XAI model **91.75%**, and the transparent router **91.50%** macro F1. The router captures `(91.50−90.34)/(91.75−90.34) = 82.3%` of the available update gain. The updated model also remains stronger on clean images, although the router becomes more conservative and records 18 harmful routes against 30 helpful routes in 2014.

### 4.1 Enhanced profile results

| Scenario | Year | Historical XAI | Enhanced XAI | Enhanced routed | Helpful | Harmful |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(enhanced_rows)}

The enhanced curriculum improved the always-new 2014 combined score from **91.75% to 91.99%**, a further **0.24 points**. The routed score increased from **91.50% to 91.85%**, a further **0.35 points**. The modest change indicates that simply increasing adaptation capacity is insufficient to create a 3-point separation.

{ablation_section}

## 5. What the saliency panels show

Each panel places the input beside the historical and updated attention maps, then prints the router features and decision. This makes a route auditable: a reviewer can compare *where* each ViT looked with the numerical evidence used by the logistic rule.

![Combined drift example from 2012](20260928T161159Z-horizon-router/combined/2012/sample_0777.png)

![Combined drift example from 2014](20260928T161159Z-horizon-router/combined/2014/sample_0777.png)

## 6. Interpretation and limits

The longer test horizon exposes a clearer decline under compound drift and a larger benefit from drift-aware updating. The rule remains explainable because its four standardized inputs, coefficients, threshold, and saliency floor are saved in `router_policy.json`. Coefficient signs are empirical correlations from 2011 calibration, not causal explanations.

The evidence is limited to one seed, one temporal dataset stream, synthetic corruptions, and one fitted threshold. No confidence intervals or multi-seed hypothesis tests are reported. The enhanced configuration produced a valid negative result for H1: the gain remained below 3 points. That target must not be presented as achieved.

The next defensible step is repeated temporal backtesting with bootstrap confidence intervals and an untouched external dataset. The router should also be compared with always-old, always-new, confidence-only, and saliency-only ablations. Report macro F1 and macro IoU by year and drift type, helpful and harmful routes, update-trigger precision/recall, throughput, and training time per epoch.

## 7. Reproducibility artifacts

- `long_horizon_summary.json`: compact protocol, scores, and router policy.
- `router_policy.json`: learned standardized logistic coefficients and thresholds.
- `20260928T161159Z-horizon-router/`: 45 saved saliency panels plus sample metadata.
- Enhanced server run: `runs/siamese/20260929-enhanced-3pp/`.
- Source checkpoints remain server-side because each model file is about 85 MB.

## References

1. Dosovitskiy et al. (2020), *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. https://arxiv.org/abs/2010.11929
2. Oquab et al. (2023), *DINOv2: Learning Robust Visual Features without Supervision*. https://arxiv.org/abs/2304.07193
3. Hendrycks and Dietterich (2019), *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. https://arxiv.org/abs/1903.12261
4. Hendrycks et al. (2019), *AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty*. https://arxiv.org/abs/1912.02781
5. Lin et al. (2021), *CLEAR: A Benchmark for Continual Learning on Real-World Imagery*. https://clear-benchmark.github.io/
"""
    PAPER_MD.write_text(markdown, encoding="utf-8")


def chart() -> Drawing:
    drawing = Drawing(170 * mm, 78 * mm)
    plot = LinePlot()
    plot.x = 15 * mm
    plot.y = 14 * mm
    plot.width = 145 * mm
    plot.height = 50 * mm
    plot.data = [
        list(zip(YEARS, RESULTS["combined"][name]))
        for name in ("baseline", "historical_xai", "updated_xai", "routed")
    ]
    palette = [colors.HexColor("#6B7280"), colors.HexColor("#2563EB"), colors.HexColor("#059669"), colors.HexColor("#D97706")]
    for i, colour in enumerate(palette):
        plot.lines[i].strokeColor = colour
        plot.lines[i].strokeWidth = 2
        plot.lines[i].symbol = None
    plot.xValueAxis.valueMin = 2012
    plot.xValueAxis.valueMax = 2014
    plot.xValueAxis.valueSteps = YEARS
    plot.yValueAxis.valueMin = 0.89
    plot.yValueAxis.valueMax = 0.97
    plot.yValueAxis.valueSteps = [0.90, 0.92, 0.94, 0.96]
    drawing.add(plot)
    labels = [("Baseline", 10), ("Historical XAI", 50), ("Updated XAI", 102), ("Routed", 145)]
    for i, (label, x) in enumerate(labels):
        drawing.add(__import__("reportlab.graphics.shapes", fromlist=["String"]).String(x * mm, 4 * mm, label, fontSize=7, fillColor=palette[i]))
    return drawing


def build_pdf() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    ablation = load_ablation()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#13293D"), alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#52616B"), alignment=TA_CENTER, spaceAfter=14))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=14, leading=17, textColor=colors.HexColor("#0B6E69"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=7))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.5, leading=10, textColor=colors.HexColor("#4B5563")))

    doc = SimpleDocTemplate(str(PAPER_PDF), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=15 * mm, title="Transparent Saliency Routing for Temporal Image Drift")
    story = [
        Paragraph("Transparent Saliency Routing<br/>for Temporal Image Drift", styles["PaperTitle"]),
        Paragraph("A compact mock research paper • fixed-horizon prototype run • 29 September 2026", styles["Subtitle"]),
        Paragraph("Abstract", styles["H1x"]),
        Paragraph("This prototype evaluates a ViT-only drift correction pipeline whose update decision can be inspected directly. Historical models train on 2004–2008, model selection uses 2009, drift adaptation uses 2010, router calibration uses 2011, and 2012–2014 remain untouched until final evaluation. Under combined drift in 2014, the transparent router improves macro F1 by 1.16 percentage points over the historical XAI model and retains 82.3% of the gain available from always choosing the updated model.", styles["Bodyx"]),
        Paragraph("1. Research question", styles["H1x"]),
        Paragraph("Can visible saliency change and prediction evidence determine when a drift-adapted ViT should replace a historical ViT prediction across a realistic future horizon?", styles["Bodyx"]),
        Paragraph("2. Architecture and transparent routing", styles["H1x"]),
        Paragraph("The experiment compares a baseline ViT, an attention-regularised XAI ViT, and a paired historical/newer XAI system. Its router is a standardized logistic rule over four logged quantities: saliency Jensen–Shannon divergence, confidence gain, entropy reduction, and class-evidence gain. A minimum saliency-change floor must also be passed. The saved coefficients and threshold make every update decision reproducible.", styles["Bodyx"]),
    ]

    architecture = [["Component", "Mechanism", "Role in drift correction"],
                    ["DINOv2-small", "Class token + patch tokens", "Shared ViT representation"],
                    ["Baseline head", "Uniform patch pooling", "Historical reference"],
                    ["XAI saliency head", "Learned scalar gate per patch", "Weights patches inside forward pass"],
                    ["Counterfactual path", "Complement of salient patches", "Tests class-evidence faithfulness"],
                    ["Twin comparison", "Frozen historical + adapted newer ViT", "Measures prediction and map change"],
                    ["Transparent router", "Four-feature logistic rule", "Chooses old or new prediction"]]
    architecture_table = Table(architecture, colWidths=[34 * mm, 59 * mm, 67 * mm], repeatRows=1)
    architecture_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13293D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [
        PageBreak(),
        Paragraph("2.1 Model mechanics", styles["H1x"]),
        Paragraph("DINOv2-small emits a class token and patch-token grid. The baseline pools patches uniformly. The XAI model learns one gate per patch, normalises the gates into a saliency distribution, and concatenates the saliency-weighted patch sum with the class token before classification. The explanation therefore participates directly in prediction.", styles["Bodyx"]),
        architecture_table,
        Paragraph("2.2 Saliency faithfulness and update loss", styles["H1x"]),
        Paragraph("A complementary path pools low-saliency patches. Training penalises the model unless selected patches provide greater true-class evidence than their complement. During adaptation, the pairwise term keeps historical and newer maps close when the historical prediction is correct, while imposing a divergence margin when it is wrong. Classification, pairwise saliency, and counterfactual evidence losses are optimized jointly.", styles["Bodyx"]),
        Paragraph("2.3 Routing equation", styles["H1x"]),
        Paragraph("For saliency JS divergence, confidence gain, entropy reduction, and evidence gain, the router computes z = (f − mean) / scale and p(update) = sigmoid(w·z + b). It selects the newer class only when the models disagree, saliency JS clears the saved floor, and p(update) clears the saved threshold. Every term is logged for sample-level audit.", styles["Bodyx"]),
    ]

    protocol = [["Stage", "Years", "Purpose"], ["Historical training", "2004–08", "Fit original ViTs"], ["Selection", "2009", "Choose checkpoints"], ["Adaptation", "2010", "Update XAI ViT with mild drift"], ["Calibration", "2011", "Fit explicit route rule"], ["Frozen test", "2012–14", "Measure future drift"]]
    table = Table(protocol, colWidths=[42 * mm, 25 * mm, 93 * mm], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13293D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    policy = json.loads((REPORTS / "router_policy.json").read_text(encoding="utf-8"))
    policy_rows = [["Transparent feature", "Coefficient"]]
    for name, coefficient in zip(policy["feature_names"], policy["coefficients"]):
        policy_rows.append([name.replace("_", " ").title(), f"{coefficient:+.3f}"])
    policy_rows += [["Route probability threshold", f"{policy['probability_threshold']:.2f}"], ["Saliency-change floor", f"{policy['saliency_floor']:.4f}"]]
    policy_table = Table(policy_rows, colWidths=[105 * mm, 45 * mm], repeatRows=1)
    policy_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B6E69")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (1, 1), (1, -1), "RIGHT"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [
        PageBreak(),
        Paragraph("3. Chronological protocol", styles["H1x"]),
        Paragraph("The split models deployment: historical learning stops before adaptation, and the future horizon is used only for final measurement. This prevents year-by-year retraining from leaking test information into the model.", styles["Bodyx"]),
        table,
        Paragraph("3.1 Optional realistic drift", styles["H1x"]),
        Paragraph("The drift toggle applies mild saturation change, Gaussian sensor noise, and partial occlusion. The newer ViT sees a deterministic 50/50 mixture of clean and individually corrupted 2010 samples. The router is calibrated on moderate individual corruptions in 2011. Combined corruption is withheld until 2012–2014, making it a composition shift rather than a memorised augmentation recipe.", styles["Bodyx"]),
        Paragraph("3.2 Saved routing policy", styles["H1x"]),
        policy_table,
        Spacer(1, 3 * mm),
        Paragraph("Coefficients act on standardized features. Their signs describe the 2011 calibration relationship; they are not causal effects. The explicit saliency floor ensures that saliency remains part of the decision rather than a decorative post-hoc plot.", styles["Smallx"]),
        PageBreak(),
        Paragraph("4. Enhanced correction experiment", styles["H1x"]),
        Paragraph("Status: completed as server run 20260929-enhanced-3pp. The 3-point value was fixed before inspection; the run did not meet that primary target.", styles["Bodyx"]),
    ]
    enhanced = [["Controlled item", "Standard profile", "Enhanced profile"],
                ["2010 stream", "50% clean; severity 1", "25% clean; individual severity 1–3"],
                ["Trainable backbone", "Final ViT block", "Final three ViT blocks"],
                ["Learning rates", "All trainable: 1e-5", "Backbone 1e-5; heads 5e-5"],
                ["Label smoothing", "0.00", "0.05"],
                ["Saliency loss weight", "0.10", "0.25"],
                ["Divergence margin", "0.05", "0.08"],
                ["Saliency floor", "Fixed median", "2011-calibrated quantile"]]
    enhanced_table = Table(enhanced, colWidths=[50 * mm, 49 * mm, 61 * mm], repeatRows=1)
    enhanced_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B6E69")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [
        enhanced_table,
        Paragraph("4.1 Fixed controls", styles["H1x"]),
        Paragraph("Historical checkpoint, CLEAR-100 chronology, random seed, epoch budget, 2011 calibration labels, frozen 2012–2014 test streams, corruption code, batch size, and metrics must match the standard run. Combined corruption remains absent from adaptation and calibration.", styles["Bodyx"]),
        Paragraph("4.2 Predeclared hypotheses", styles["H1x"]),
        Paragraph("H1 not met: enhanced XAI gained 1.40 points over baseline ViT on 2014 combined drift. H2 met: routing retained 91.5% of the newer model's gain over historical XAI. H3 met: routed clean 2014 macro F1 was 0.06 points above historical XAI. Test labels must not be used for another tuning cycle.", styles["Bodyx"]),
        Paragraph("4.3 Evaluation matrix", styles["H1x"]),
        Paragraph("The selected epoch was 4, with 94.39% calibration macro F1; epoch 5 fell to 93.00%. Post-warm-up epochs took about 138 seconds. Final reporting includes accuracy, macro F1, macro IoU, update-trigger metrics, helpful and harmful routes, timing, and saliency-map overlap. Multiple seeds remain necessary before a final claim.", styles["Bodyx"]),
        PageBreak(),
        Paragraph("5. Completed fixed-horizon results", styles["H1x"]),
        Paragraph("Macro F1 under combined saturation, noise, and occlusion. The widening decline across years makes the temporal effect visible.", styles["Bodyx"]),
        chart(),
    ]

    data = [["Year", "Baseline", "Hist. XAI", "Updated XAI", "Routed", "Helpful / harmful"]]
    for i, year in enumerate(YEARS):
        r = RESULTS["combined"]
        data.append([str(year), pp(r["baseline"][i]), pp(r["historical_xai"][i]), pp(r["updated_xai"][i]), pp(r["routed"][i]), f"{r['helpful_routes'][i]} / {r['harmful_routes'][i]}"])
    results_table = Table(data, colWidths=[18 * mm, 24 * mm, 27 * mm, 29 * mm, 23 * mm, 39 * mm], repeatRows=1)
    results_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B6E69")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [results_table, Spacer(1, 4 * mm), Paragraph("In 2014, the updated XAI model gains 1.41 points over historical XAI. Routing retains 1.16 points: (91.50−90.34)/(91.75−90.34) = 82.3% of the available gain. Clean-data macro F1 is 95.13% baseline, 95.24% historical XAI, 95.85% updated XAI, and 95.46% routed.", styles["Bodyx"])]

    enhanced_data = [["Year", "Hist. XAI", "Std. updated", "Enhanced updated", "Enhanced routed", "Helpful / harmful"]]
    for i, year in enumerate(YEARS):
        enhanced_data.append([str(year), pp(ENHANCED["combined"]["old"][i]),
                              pp(RESULTS["combined"]["updated_xai"][i]),
                              pp(ENHANCED["combined"]["new"][i]),
                              pp(ENHANCED["combined"]["routed"][i]),
                              f"{ENHANCED['combined']['helpful_routes'][i]} / {ENHANCED['combined']['harmful_routes'][i]}"])
    enhanced_results_table = Table(enhanced_data, colWidths=[16 * mm, 24 * mm, 28 * mm, 34 * mm, 32 * mm, 34 * mm], repeatRows=1)
    enhanced_results_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13293D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [Paragraph("5.1 Enhanced combined-drift result", styles["H1x"]), enhanced_results_table,
              Spacer(1, 3 * mm),
              Paragraph("In 2014, enhanced XAI reaches 91.99%: +1.40 points over baseline ViT and +1.65 over historical XAI. Enhanced routing reaches 91.85%, retaining 91.5% of the available gain. Relative to the standard profile, enhancement adds 0.24 points always-new and 0.35 points routed. H1 fails; H2 and H3 pass.", styles["Bodyx"])]

    if ablation:
        runs = ablation["runs"]
        ablation_data = [["Five-epoch configuration", "Clean routed", "Combined new", "Combined routed", "Helpful / harmful"]]
        for key, label in (("matched_standard", "Matched standard"),
                           ("enhanced_full", "Enhanced + auxiliary XAI"),
                           ("enhanced_no_aux", "Enhanced, no auxiliary XAI")):
            clean = runs[key]["scenarios"]["clean"]["2014"]
            combined = runs[key]["scenarios"]["combined"]["2014"]
            ablation_data.append([label, pp(clean["routed"]), pp(combined["new"]), pp(combined["routed"]),
                                  f"{combined['helpful_routes']} / {combined['harmful_routes']}"])
        ablation_table = Table(ablation_data, colWidths=[50 * mm, 25 * mm, 28 * mm, 31 * mm, 34 * mm], repeatRows=1)
        ablation_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7C3AED")),
                                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                                            ("FONTSIZE", (0, 0), (-1, -1), 7.4),
                                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F3FF")]),
                                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        full = runs["enhanced_full"]["scenarios"]["combined"]["2014"]
        no_aux = runs["enhanced_no_aux"]["scenarios"]["combined"]["2014"]
        story += [Paragraph("5.2 Controlled five-epoch ablation", styles["H1x"]), ablation_table,
                  Paragraph(f"The enhanced auxiliary saliency objectives changed 2014 combined macro F1 by {(full['new'] - no_aux['new']) * 100:+.2f} points for always-new inference and {(full['routed'] - no_aux['routed']) * 100:+.2f} points after routing. The control retains the saliency-gated forward pass and removes only the pairwise divergence and complementary-evidence objectives.", styles["Bodyx"])]

    panels = [RUN / "combined" / "2012" / "sample_0777.png", RUN / "combined" / "2014" / "sample_0777.png"]
    story += [PageBreak(), Paragraph("6. Saliency audit examples", styles["H1x"]), Paragraph("Each panel shows the clean reference, drifted input, historical saliency, and updated saliency. Together with the saved decision log, this lets a reviewer connect the route to visible attention changes and prediction evidence.", styles["Bodyx"])]
    for year, panel in zip((2012, 2014), panels):
        if panel.exists():
            story += [Paragraph(f"Combined drift • {year}", styles["Smallx"]), Image(str(panel), width=155 * mm, height=87 * mm), Spacer(1, 3 * mm)]
    if SALIENCY_ANALYSIS_PNG.exists():
        story += [PageBreak(), Paragraph("6.1 Saliency features by routing outcome", styles["H1x"]),
                  Paragraph("The box plots group the 2014 combined-drift decisions as helpful, harmful, neutral routed, or kept historical. They test whether visible saliency divergence and the explicit route probability separate useful from harmful updates; they do not establish that saliency caused the correction.", styles["Bodyx"]),
                  Image(str(SALIENCY_ANALYSIS_PNG), width=168 * mm, height=67 * mm)]
    story += [PageBreak(), Paragraph("7. Interpretation", styles["H1x"]), Paragraph("The completed enhanced run improves both the always-new and routed combined-drift scores, but only modestly. This indicates that broader corruption exposure, three trainable blocks, and stronger saliency losses do not by themselves create the targeted 3-point separation. The routed model preserves most of the newer model's benefit, showing that transparent selection remains viable even though the underlying adaptation gain is limited.", styles["Bodyx"]), Paragraph("8. Limitations and validity threats", styles["H1x"]), Paragraph("Both completed profiles use one seed on one temporal stream with synthetic corruptions and one calibrated threshold. They have no confidence intervals or external validation dataset. H1 is a valid negative result. Increased optimization capacity and the saliency curriculum changed together, so ablations must isolate trainable blocks, corruption curriculum, saliency loss, and router calibration before attributing the incremental gain to saliency.", styles["Bodyx"]), Paragraph("9. Next experimental sequence", styles["H1x"]), Paragraph("Freeze these outputs, compute paired per-sample bootstrap intervals, and repeat both profiles across at least three seeds. Compare always-old, always-new, confidence-only, saliency-only, and full routing. A stronger next method should introduce saliency-guided patch corruption or clean-drift consistency training as a predeclared intervention, then repeat on a second temporal image dataset.", styles["Bodyx"]), Paragraph("10. Reproducibility", styles["H1x"]), Paragraph("The repository includes compact JSON results, learned router policy, 45 saliency panels, report generator, and all training source. Large checkpoints remain server-side. Completed run identifiers include baseline 20260928T152657Z-horizon, standard router 20260928T161159Z-horizon-router, and enhanced router 20260929-enhanced-3pp.", styles["Bodyx"]), Paragraph("References", styles["H1x"]), Paragraph("[1] Dosovitskiy et al., Vision Transformer, 2020. [2] Oquab et al., DINOv2, 2023. [3] Hendrycks & Dietterich, ImageNet-C, 2019. [4] Hendrycks et al., AugMix, 2019. [5] Lin et al., CLEAR, 2021. Full links appear in the Markdown version.", styles["Smallx"])]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(18 * mm, 9 * mm, "Rik/FYP • transparent router prototype")
        canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def clean_large_inputs() -> None:
    for name in ("baseline_future_drift.json", "xai_future_drift.json", "router_future_drift.json"):
        (REPORTS / name).unlink(missing_ok=True)


if __name__ == "__main__":
    build_summary()
    build_markdown()
    build_pdf()
    clean_large_inputs()
    print(PAPER_MD)
    print(PAPER_PDF)
    print(SUMMARY_JSON)
