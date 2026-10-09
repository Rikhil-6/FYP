"""Expand the supplied thesis draft without changing its page setup or figure styling."""
from pathlib import Path
from typing import Iterable

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/docx/updated_ViT_router_mock_paper_2_router_plot.docx"
OUTPUT = ROOT / "output/docx/updated_ViT_router_mock_paper_2_literature_revision.docx"


def find_paragraph(doc: Document, prefix: str):
    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise ValueError(f"Paragraph not found: {prefix}")


def replace_text(paragraph, text: str, bold_prefix: str | None = None):
    for run in paragraph.runs:
        run._element.getparent().remove(run._element)
    if bold_prefix and text.startswith(bold_prefix):
        paragraph.add_run(bold_prefix).bold = True
        paragraph.add_run(text[len(bold_prefix):])
    else:
        paragraph.add_run(text)


def remove_paragraph(paragraph):
    element = paragraph._element
    element.getparent().remove(element)


def insert_paragraphs(anchor, entries: Iterable[tuple[str, str]]):
    for style, text in entries:
        paragraph = anchor.insert_paragraph_before(style=style)
        paragraph.add_run(text)


def shade_cell(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)


def insert_table_before(doc: Document, anchor, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = doc.tables[0].style
    table.autofit = True
    header = table.rows[0]
    repeat_table_header(header)
    for index, value in enumerate(headers):
        cell = header.cells[index]
        cell.text = value
        shade_cell(cell, "D9EAF7")
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(8)
    for row_values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row_values):
            cells[index].text = value
            for paragraph in cells[index].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(8)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = width
    anchor._p.addprevious(table._tbl)
    return table


def renumber_heading(doc: Document, old: str, new: str):
    paragraph = next((item for item in doc.paragraphs if item.text.strip() == old), None)
    if paragraph is None:
        raise ValueError(f"Heading not found: {old}")
    replace_text(paragraph, new)


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    doc = Document(SOURCE)
    normal_style = find_paragraph(doc, "Image classification systems").style.name
    heading_1 = find_paragraph(doc, "1. Introduction").style.name
    heading_2 = find_paragraph(doc, "2.1 ViT classifier").style.name

    # 1. Introduction: separate the problem, questions, and contribution statement.
    problem = find_paragraph(doc, "Image classification systems")
    problem.insert_paragraph_before("1.1 Operational problem and thesis claim", style=heading_2)
    thesis = find_paragraph(doc, "Updating the model is an obvious response")
    replace_text(
        thesis,
        "This thesis investigates a controlled alternative to wholesale model replacement. "
        "A frozen historical Vision Transformer (ViT) is retained as a behavioural reference, "
        "a second ViT is adapted on a later time window, and a transparent router decides which "
        "prediction to use for each image. The central claim is deliberately narrower than a claim "
        "of universal drift correction: when covariate drift changes where a ViT finds class evidence, "
        "an explanation-aware comparison can preserve much of the adapted model's benefit while making "
        "the cost of harmful replacements visible. The claim is tested through chronological holdouts, "
        "source-separated stress tests, matched ablations, and per-image route audits."
    )
    remove_paragraph(find_paragraph(doc, "The saliency map is part of the model"))
    remove_paragraph(find_paragraph(doc, "The paired comparison follows the broad idea"))

    research_intro = find_paragraph(doc, "This thesis is examined through three research questions")
    replace_text(research_intro, "1.2 Research questions and measurable objectives")
    research_intro.style = heading_2
    find_paragraph(doc, "1. RQ1:").insert_paragraph_before(
        "The objective is to measure both predictive improvement and update safety. Macro F1 and macro "
        "IoU quantify class-balanced classification performance; retained gain measures how much of the "
        "always-updated improvement survives routing; and helpful, harmful, and missed switches expose the "
        "decision trade-off hidden by an aggregate score. A three-point macro-F1 gain was predeclared as a "
        "stretch target for the combined condition, while clean degradation was limited to 0.50 points.",
        style=normal_style,
    )
    contribution = find_paragraph(doc, "The experiments use only ViT models")
    contribution.insert_paragraph_before("1.3 Contributions and scope", style=heading_2)
    replace_text(
        contribution,
        "The work contributes a ViT-only temporal prototype with an embedded patch-evidence gate, an "
        "asymmetric historical-versus-adapted comparison, an explicit four-feature routing policy, and a "
        "logged audit of route outcomes and saliency deletion behaviour. Its intended contribution is a "
        "reproducible decision framework and an analysis of where it succeeds or fails. It does not claim "
        "that saliency proves causality, that a single CLEAR-100 stream establishes generalisation, or that "
        "the router is universally preferable to always using the updated model."
    )

    # 2. Literature review and design rationale.
    system_anchor = find_paragraph(doc, "2. System design")
    literature_entries = [
        (heading_1, "2. Literature review and design rationale"),
        (heading_2, "2.1 Responses to distribution shift"),
        (normal_style,
         "Research on distribution shift spans several intervention levels. Robust training and corruption "
         "augmentation attempt to make the original model less sensitive before deployment [1, 17]. Test-time "
         "adaptation instead changes the deployed model using target batches: Tent minimises prediction entropy "
         "through normalization parameters [20], CoTTA uses averaged predictions and partial source-weight "
         "restoration to reduce error accumulation in changing environments [21], and EATA filters uninformative "
         "samples while regularising important weights to limit forgetting [22]. These methods are valuable when "
         "labels and source data are unavailable at deployment, but they require online optimisation and can still "
         "degrade in-distribution behaviour. This project studies a different operational point: a labelled update "
         "window is available, the historical model must remain recoverable, and every model switch should be auditable."),
        (normal_style,
         "Continual-learning work addresses longer sequences through replay, regularisation, parameter-efficient "
         "updates, and representation correction. Wu et al. combine LoRA with mean-shift compensation, covariance "
         "calibration, and patch-token distillation to reduce semantic drift in class-incremental ViTs [24]. Such "
         "methods update a continuing learner and preserve old classes. The present study keeps class definitions "
         "fixed and instead asks whether two time-indexed models can be compared safely at inference. SelectiveNet's "
         "risk-coverage formulation is therefore closer to the decision problem [23], although the router here "
         "defers to another classifier rather than abstaining."),
        (heading_2, "2.2 From post-hoc maps to saliency-guided learning"),
        (normal_style,
         "A post-hoc heatmap can describe a prediction without changing it. The stronger claim examined here is "
         "that explanation structure can participate in learning and model selection. SGAN constrains self-attention "
         "with saliency priors to reduce spread into irrelevant background [4]. XGD transfers class predictions and "
         "saliency-refined features, with ablations showing that the saliency term contributes separately from ordinary "
         "distillation [5]. Ismail et al. train models to preserve predictions after masking low-saliency features, "
         "linking the map to a behavioural perturbation [7]. Saliency Guided Image Warping uses source saliency to "
         "control which regions dominate domain alignment [8], while SGT removes low-saliency ViT patches to reduce "
         "background shortcuts [25]. Together these papers motivate using patch weights in the forward computation and "
         "checking them by deletion, while also warning that a cleaner map does not guarantee higher accuracy."),
        (heading_2, "2.3 Why an asymmetric Siamese ViT comparison"),
        (normal_style,
         "Siamese architectures are useful when the decision depends on a relation between two observations rather "
         "than on either observation alone. Siamese-ViT shows that paired transformer branches can compare global and "
         "local evidence across views [9]. Yang et al. show that strict weight symmetry can be inappropriate when the "
         "two temporal inputs have different roles or distributions [10]. The analogy in this project is functional: "
         "the two branches share the same DINOv2-small architecture and initialization, but the historical branch is "
         "frozen while the newer branch is adapted. This holds backbone family constant and makes model age, training "
         "window, and evidence movement the variables under study."),
        (normal_style,
         "The implementation is more accurately described as an asymmetric Siamese evaluation architecture than as "
         "a fully end-to-end Siamese metric network. It does not learn a hidden pair embedding or distance function. "
         "Instead, both ViTs expose predictions and normalized patch weights, and a logistic policy combines four named "
         "differences. This sacrifices some routing capacity compared with a learned mixture-of-experts gate, but it "
         "allows each switch to be reconstructed from stored coefficients and features. Always-new inference remains "
         "the performance ceiling; the router is justified only when auditability or avoidance of harmful updates is an "
         "operational requirement."),
        (heading_2, "2.4 Quantitative context for expected improvements"),
        (normal_style,
         "Reported gains from saliency guidance and temporal correction vary widely with task, shift severity, and metric. "
         "The comparisons below are used to set expectations rather than to claim direct equivalence. Dense prediction, "
         "retrieval, and class-incremental accuracy are not interchangeable with CLEAR-100 macro F1."),
    ]
    insert_paragraphs(system_anchor, literature_entries)

    literature_rows = [
        ("SGAN [4]", "Weakly supervised segmentation", "55.0→64.2 mIoU (+9.2)", "Strong saliency prior; different task and supervision"),
        ("XGD [5]", "Teacher-student segmentation", "65.25→69.82 mIoU (+4.57); saliency-only +3.52", "Supports explicit saliency ablation"),
        ("Saliency warping [8]", "Unsupervised domain adaptation", "+3.0 to +6.3 on several shifts; +0.6 on DarkZurich", "Effects depend strongly on target domain"),
        ("SGT [25]", "ViT shortcut reduction", "INbreast +2.41 accuracy, +3.06 F1; FIGRIM −0.32 accuracy", "Interpretability may improve without accuracy gain"),
        ("Semantic-drift ViT [24]", "Class-incremental learning", "79.36→81.88 accuracy (+2.52)", "Component increments include +0.28 for patch distillation"),
        ("Siamese-ViT [9]", "Cross-view retrieval", "+2.33 Recall@1 and +1.38 AP vs TransGeo", "Supports paired ViT comparison, not temporal routing"),
        ("This project", "Temporal classification", "+3.45 routed under motion blur; +2.03 under mixed drift", "Single seed; auxiliary saliency effect +0.90/+0.84"),
    ]
    insert_table_before(doc, system_anchor,
                        ("Study", "Setting", "Reported absolute change", "Interpretation for this thesis"),
                        literature_rows)
    system_anchor.insert_paragraph_before(
        "The table suggests two defensible expectations. First, a three-point target is ambitious but plausible under a "
        "shift that creates substantial headroom; it is not a minimum effect that every drift source should meet. Second, "
        "individual components often contribute less than the complete method. The project therefore reports percentage-"
        "point changes beside raw scores, separates updated-model gain from routing retention, and treats sub-point "
        "saliency effects as evidence that requires repeated seeds rather than as proof of a general mechanism.",
        style=normal_style,
    )
    system_anchor.insert_paragraph_before("2.5 Design decisions derived from the literature", style=heading_2)
    decision_rows = [
        ("DINOv2-small ViT", "Strong pretrained patch and class tokens [13, 14]", "Same backbone in both branches; final three blocks updated", "Future-year macro F1 and training cost"),
        ("Embedded saliency gate", "Trainable saliency and perturbation evidence [4, 5, 7, 25]", "Patch weights form the classified representation", "Ablation gain and deletion margin"),
        ("Asymmetric pair", "Paired ViTs and unequal temporal roles [9, 10]", "Frozen historical reference versus adapted candidate", "Helpful, harmful, and missed switches"),
        ("Explicit logistic router", "Interpretable policies and selective risk [15, 23]", "Four stored features plus thresholds", "Retained gain and harmful-route count"),
        ("Frozen future horizon", "Temporal benchmarks and anti-forgetting concerns [2, 21, 22]", "No updates on 2012–2014", "Year-specific clean and drift scores"),
        ("Source-separated stressors", "Corruption and natural-shift evidence [1, 3, 8, 19]", "Optical, sensor, digital, geometric, and mixed shifts", "Effect size by source"),
    ]
    insert_table_before(doc, system_anchor,
                        ("Choice", "Rationale", "Implementation", "Evidence required"), decision_rows)
    system_anchor.insert_paragraph_before(
        "These choices make the prototype feasible on a single modest GPU and keep the decision trail compact. They do "
        "not establish superiority over end-to-end selective networks or continual test-time adaptation. The comparison "
        "is strongest when the requirement is a reviewable transition between two versioned ViTs with labelled update data.",
        style=normal_style,
    )

    # Renumber the original material after inserting the literature review.
    renumber_heading(doc, "2. System design", "3. System design")
    renumber_heading(doc, "2.1 ViT classifier and embedded explanation", "3.1 ViT classifier and embedded explanation")
    renumber_heading(doc, "2.2 Asymmetric Siamese comparison and explicit routing rule", "3.2 Asymmetric Siamese comparison and explicit routing rule")
    renumber_heading(doc, "3. Temporal protocol", "4. Temporal protocol")
    renumber_heading(doc, "3.1 Enhanced correction experiment", "4.1 Enhanced correction experiment")
    renumber_heading(doc, "3.2 Hypotheses and success criteria", "4.2 Hypotheses and success criteria")
    renumber_heading(doc, "3.3 Enhanced result against the hypotheses", "4.3 Enhanced result against the hypotheses")
    renumber_heading(doc, "3.4 Checkpoint and interruption recovery", "4.4 Checkpoint and interruption recovery")
    renumber_heading(doc, "4. Main results", "5. Main results")
    renumber_heading(doc, "4.1 Enhanced profile results", "5.1 Enhanced profile results")
    renumber_heading(doc, "4.2 Controlled five-epoch ablation", "5.2 Controlled five-epoch ablation")
    renumber_heading(doc, "4.3 Source-separated drift evaluation", "5.3 Source-separated drift evaluation")
    renumber_heading(doc, "5. Saliency and routing analysis", "6. Saliency and routing analysis")
    renumber_heading(doc, "5.1 Quantitative saliency outcome analysis", "6.1 Quantitative saliency outcome analysis")
    renumber_heading(doc, "5.2 Counterexample and faithfulness audit", "6.2 Counterexample and faithfulness audit")
    renumber_heading(doc, "5.3 Scope of the method", "6.3 Scope of the method")
    renumber_heading(doc, "6. Conclusion", "7. Discussion, limitations, and conclusion")

    for paragraph in doc.paragraphs:
        if "results appear in Section 4.3" in paragraph.text:
            replace_text(paragraph, paragraph.text.replace("Section 4.3", "Section 5.3"))

    # The supplied file contains a manual break before the original Section 2.1.
    # After inserting the review it leaves an almost empty page, so remove only
    # that break while preserving all other pagination from the supplied file.
    paragraphs = doc.paragraphs
    for index, paragraph in enumerate(paragraphs[1:-1], 1):
        if (not paragraph.text.strip() and paragraph._p.xpath(".//w:br")
                and paragraphs[index - 1].text.startswith("The baseline and historical XAI models")
                and paragraphs[index + 1].text.startswith("The three classifier views")):
            remove_paragraph(paragraph)
            break

    # Clarify what the existing numbers represent and record the implemented protocol correction.
    enhanced_result = find_paragraph(doc, "The completed enhanced run reused")
    replace_text(
        enhanced_result,
        "The completed enhanced run used the same historical checkpoint, seed 42, and frozen 2012–2014 evaluation "
        "streams as its controls. Its adapted checkpoint and routing policy were both chosen using the 2011 sample. "
        "Epoch 4 was selected by 2011 macro F1 (94.39%); epoch 5 declined to 93.00%, indicating overfitting after the "
        "selected checkpoint. Training took approximately 138 seconds per epoch after warm-up. Combined corruption "
        "remained unavailable during adaptation and calibration. Because checkpoint selection and policy fitting share "
        "2011 labels, all reported numerical results are exploratory rather than confirmatory."
    )
    protocol_correction_anchor = find_paragraph(doc, "4.2 Hypotheses")
    insert_paragraphs(protocol_correction_anchor, [
        (heading_2, "4.1.1 Protocol correction for future runs"),
        (normal_style,
         "Reusing 2011 for both checkpoint selection and router calibration gives the policy information about the sample "
         "that already determined which adapted model survived. This can make route thresholds look better matched to that "
         "window than they would be on new data. The code now divides 2011 deterministically within each class: one partition "
         "selects the adapted checkpoint and the other fits logistic coefficients, the saliency floor, and the probability "
         "threshold. The default split is 50/50 and is recorded with both sample counts. A legacy flag reproduces the shared "
         "protocol for audit purposes. The tables in this paper were not recomputed after that correction, so the distinction "
         "is explicit rather than retroactively changing the meaning of the reported runs."),
    ])

    # Replace the short conclusion with a structured interpretation.
    conclusion = find_paragraph(doc, "The longer horizon makes the decline")
    conclusion.insert_paragraph_before("7.1 What the experiment establishes", style=heading_2)
    replace_text(
        conclusion,
        "The experiment establishes that update value depends on the drift source. Routing improves over the frozen historical "
        "ViT in all eleven 2014 conditions, but the magnitude ranges from +0.11 points under illumination to +3.45 under motion "
        "blur. Under motion blur and the realistic mixture, routing retains 87.1% and 91.9% of the always-updated gain. This is "
        "evidence that a transparent branch-selection rule can preserve most of an adapted ViT's benefit when the historical "
        "model has genuinely degraded. It is not evidence that routing beats always using the updated model; the updated branch "
        "remains 0.06–0.51 points higher across the source-separated tests."
    )
    limitations = find_paragraph(doc, "The main limitations are one random seed")
    limitations.insert_paragraph_before("7.2 Negative results and failure modes", style=heading_2)
    insert_paragraphs(limitations, [
        (normal_style,
         "The predeclared three-point target was not met on the original combined condition: the enhanced updated model gained "
         "1.40 points over the baseline and routed inference gained 1.51 points over historical XAI. In the matched ablation, "
         "the explicit auxiliary saliency objectives added only +0.23 routed points on combined drift and reduced clean routed "
         "performance by 0.18 points. These outcomes show that increasing adaptation capacity and encouraging map movement do "
         "not automatically produce a large or uniformly safe correction."),
        (normal_style,
         "The route audit explains part of the gap. Helpful and harmful decisions overlap in saliency divergence and route "
         "probability, so the four features do not cleanly identify correctness. A map can change because the updated model has "
         "found better evidence, because it has shifted to a different shortcut, or because blur has redistributed all patch "
         "weights. The logistic router is deliberately simple and may underfit these interactions; a more expressive gate may "
         "improve accuracy but would weaken the direct coefficient-level explanation that motivates the design."),
        (heading_2, "7.3 Validity threats and implemented correction"),
    ])
    replace_text(
        limitations,
        "The evidence is limited by one random seed, one temporal dataset, synthetic stressors layered over natural chronology, "
        "and one fitted policy. The reported runs also reuse 2011 labels for adapted-checkpoint selection and router calibration. "
        "That reuse is a material selection-bias risk and has been corrected in the code through disjoint class-stratified "
        "partitions, but the existing figures have not been relabelled as corrected results. No confidence intervals or external "
        "validation dataset are available, and deletion creates out-of-distribution images. Saliency maps should therefore be "
        "interpreted as evidence-allocation diagnostics rather than causal or spatial ground truth."
    )
    next_experiment = find_paragraph(doc, "The next experiment should repeat")
    next_experiment.insert_paragraph_before("7.4 Feasible next evaluation", style=heading_2)
    replace_text(
        next_experiment,
        "A confirmatory run should use the corrected 2011 split, at least three matched seeds, and paired per-image bootstrap "
        "intervals. It should compare confidence-only, saliency-only, combined, and end-to-end selective gates at matched route "
        "coverage, while retaining always-historical and always-updated baselines. A random-patch intervention should accompany "
        "saliency-guided deletion, and a second temporal image dataset should test transfer. These additions are feasible because "
        "the existing run store already records per-image decisions, timing, checkpoints, and source-specific scenarios."
    )
    appendix = find_paragraph(doc, "Appendix A.")
    appendix.insert_paragraph_before("7.5 Conclusion", style=heading_2)
    appendix.insert_paragraph_before(
        "The thesis therefore supports a qualified conclusion. An asymmetric historical-versus-adapted ViT pair is a useful "
        "way to frame model updates when traceability matters, and embedded saliency supplies inspectable evidence that can "
        "influence both learning and routing. The current router improves over the historical model most clearly under motion "
        "and mixed acquisition drift, but its saliency-specific contribution is modest and source dependent. The method is best "
        "presented as a feasible, auditable prototype whose strongest value is exposing update decisions and failure modes, "
        "pending corrected multi-seed validation.",
        style=normal_style,
    )

    # Protocol table: preserve the archived-result meaning while recording the new default.
    for table in doc.tables:
        if table.cell(0, 0).text.strip() == "Stage" and table.cell(0, 1).text.strip() == "Years":
            for row in table.rows:
                if row.cells[0].text.strip() == "Router calibration":
                    row.cells[2].text = "Archived run: select adapted checkpoint and fit router on shared labels; corrected code uses disjoint class-stratified partitions"
            break

    # Add the sources used for the expanded review.
    references = next(paragraph for paragraph in doc.paragraphs
                      if paragraph.text.strip().startswith("19."))
    new_references = [
        "20. Wang et al. (2021), Tent: Fully Test-Time Adaptation by Entropy Minimization. International Conference on Learning Representations. https://openreview.net/forum?id=uXl3bZLkr3c",
        "21. Wang et al. (2022), Continual Test-Time Domain Adaptation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7201–7211. https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html",
        "22. Niu et al. (2022), Efficient Test-Time Model Adaptation without Forgetting. Proceedings of the 39th International Conference on Machine Learning, 16888–16905. https://proceedings.mlr.press/v162/niu22a.html",
        "23. Geifman and El-Yaniv (2019), SelectiveNet: A Deep Neural Network with an Integrated Reject Option. Proceedings of the 36th International Conference on Machine Learning, 2151–2159. https://proceedings.mlr.press/v97/geifman19a.html",
        "24. Wu et al. (2025), Navigating Semantic Drift in Task-Agnostic Class-Incremental Learning. https://arxiv.org/abs/2503.11756",
        "25. Ma et al. (2022), Rectify ViT Shortcut Learning by Visual Saliency. https://arxiv.org/abs/2206.08567",
    ]
    for reference in new_references:
        paragraph = references._parent.add_paragraph(style=references.style)
        paragraph.add_run(reference)
        references._p.addnext(paragraph._p)
        references = paragraph

    # Keep body paragraphs left aligned and let existing page/section settings survive unchanged.
    for paragraph in doc.paragraphs:
        if paragraph.style.name == normal_style and paragraph.alignment is None:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
