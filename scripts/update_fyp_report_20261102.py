"""Revise the report draft to align its claims with the completed evidence."""

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "saliency_guided_vit_drift_routing_revised_v2.docx"
OUTPUT = ROOT / "paper" / "FYP_report_revised_20261102.docx"


def set_paragraph(paragraph, text):
    paragraph.clear()
    paragraph.add_run(text)


def insert_after(paragraph, text, style="Normal"):
    new_p = paragraph._parent.add_paragraph(style=style)
    new_p.add_run(text)
    paragraph._p.addnext(new_p._p)
    return new_p


def insert_table_before(document, before_paragraph, rows):
    table = document.add_table(rows=1, cols=len(rows[0]))
    table.style = "TableNormal"
    for cell, value in zip(table.rows[0].cells, rows[0]):
        cell.text = value
    for values in rows[1:]:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = value
    widths = [Inches(0.55), Inches(1.05), Inches(1.25), Inches(1.0), Inches(2.1)]
    for row_index, row in enumerate(table.rows):
        for index, cell in enumerate(row.cells):
            cell.width = widths[index]
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(1)
                for run in paragraph.runs:
                    run.font.size = Pt(8)
                    if row_index == 0:
                        run.bold = True
    before_paragraph._p.addprevious(table._tbl)
    return table


def main():
    doc = Document(SOURCE)
    # Cover and title block.
    set_paragraph(doc.paragraphs[0], "Saliency as an Operational Signal for Drift-Aware Vision Model Updating")
    set_paragraph(doc.paragraphs[1], "Evidence from CLEAR-100 and Wild-Time Yearbook")

    # Replace the complete abstract, keeping its original heading and paragraph style.
    abstract = next(p for p in doc.paragraphs if p.text.startswith("This study examines whether Vision Transformer"))
    set_paragraph(abstract,
        "This study tests whether saliency can help correct predictive errors under image distribution shift, rather than serving only as a post-hoc display. On CLEAR-100, frozen DINOv2 features detect the controlled acquisition changes, while the sampled clean-year comparison does not show reliable separation. The main archived CLEAR comparison reports 82.75% macro F1 for saliency-guided updating, 81.55% for conventional fine-tuning, and 79.42% for the shared historical reference on the 2014 evidence-ranked mixture. A corrected-split, single-seed run reverses that comparison: the saliency update scores 80.78%, 0.77 points below conventional fine-tuning. A shared-checkpoint patch-weight intervention also found near-identical predictions after learned weights were replaced by uniform, shuffled, or inverted weights. A separate chronological Wild-Time Yearbook pilot found that saliency-mask LoRA exceeded an equal-area random-mask control by a mean 0.90 macro-F1 points across three seeds under an injected center occlusion; it exceeded ordinary LoRA in two seeds. It did not improve results under blur or noise, and the natural Yearbook chronology did not harm the historical model in this sample. The evidence therefore supports a narrow, exploratory occlusion-robustness signal, not a general saliency-specific correction of natural temporal drift. The paired router adds an inspectable fallback but has not shown reliable selection value beyond confidence. A planned iWildCam time-by-camera replication remains pending; this draft reports no iWildCam results.")

    # Introduction, questions, and contribution claims.
    p = next(p for p in doc.paragraphs if p.text.startswith("The study has three experiments."))
    set_paragraph(p,
        "The study combines four distinct evaluations. First, it tests unlabelled feature drift with MMD and a domain classifier, separating clean-year comparisons from controlled image changes. Second, it compares the archived CLEAR update strategies and reports the later corrected-split and shared-checkpoint controls. Third, it measures selective routing separately from always-updated accuracy. Fourth, it tests a saliency-mask LoRA intervention on chronological Yearbook data and on fixed, injected blur, occlusion, and noise. These evaluations distinguish measurable feature change, task harm, and the value of a saliency intervention; none is treated as proof of another.")
    p = next(p for p in doc.paragraphs if p.text.startswith("The three experiments use different evidence."))
    set_paragraph(p,
        "The research questions use separate evidence. RQ1 tests whether unlabelled samples show measurable feature-distribution change and whether that change coincides with a task-performance drop. RQ2 asks whether saliency-guided adaptation improves over ordinary fine-tuning and matched random controls, with clean performance reported alongside shifted performance. RQ3 asks whether the router improves decisions over always-updated and confidence-only policies. The 2.00-point gain, 75% retention, and 0.50-point clean-loss values in the original CLEAR audit were chosen after initial results as descriptive engineering references; they are not prospective hypotheses or safety standards.")
    p = next(p for p in doc.paragraphs if p.text.startswith("The contributions follow the three research questions."))
    set_paragraph(p,
        "The report contributes a reproducible drift-screening protocol, an audit of the original saliency-weighted ViT and paired router, and a matched saliency-mask adaptation pilot on chronological images. The component tests include uniform, shuffled, and inverted pooling weights on a fixed checkpoint, as well as equal-area random masking during LoRA adaptation. The results include both positive and negative outcomes: saliency masking shows a small repeated advantage over random masking under one injected occlusion, while ordinary LoRA is better on clean data and saliency does not help consistently under blur or noise. The router remains a secondary mechanism whose incremental decision value is not established.")

    # Operational scope and dataset wording.
    p = next(p for p in doc.paragraphs if p.text.startswith("Supplementary datasets assess the extension capability"))
    set_paragraph(p,
        "Supplementary datasets test specific transfer boundaries. Wild-Time Yearbook provides a low-resolution chronological portrait stream [3]; CIFAR-10.1 provides an independently recollected test set [31]; and Camelyon17-WILDS tests hospital acquisition shift [2, 32]. FMoW-WILDS and iWildCam-WILDS offer useful time-and-location metadata for future satellite and camera-trap studies. The iWildCam time-by-camera replication remains planned and has not produced results for this draft. Each benchmark requires its own chronology, group constraints, and drift verification; results are not pooled across datasets.")

    # Append a distinct Yearbook method to the existing cross-dataset protocol.
    p = next(p for p in doc.paragraphs if p.text.startswith("The three screens are feasibility experiments"))
    p = insert_after(p,
        "A separate Yearbook saliency-mask pilot uses native time buckets and split roles: the historical head trains on 1930-1989 train images and is selected on 1980-1989 test images; adaptation uses 1990-1999 train images; epoch selection uses 2000-2003 test images; and final evaluation uses 2004-2013 test images. The future test contains 335 images, all available under the configured sample limit. The three adaptation arms use the same historical initialization, rank-4 query/value LoRA modules, 1,000 adaptation images, and two epochs: ordinary LoRA, equal-area random masking plus masked-view classification loss, and saliency-ranked masking plus the same loss. The two masking arms have 19,202 trainable parameters and a masked-view loss weight of 0.5. A frozen historical model estimates patch saliency from its predicted class; future labels are never used to form the masks or choose epochs. This screen is separate from the previous Yearbook frozen-head comparison.")
    p = next(p for p in doc.paragraphs if p.text.startswith("Reusing 2011 for both checkpoint selection"))
    set_paragraph(p,
        "Reusing 2011 for both checkpoint selection and router calibration gives the policy information about the sample that already determined which adapted model survived. The code divides 2011 deterministically within each class: one partition selects the adapted checkpoint and the other fits the logistic coefficients, saliency floor, and routing threshold. The default split is 50/50 and is recorded with both sample counts. A legacy flag reproduces the shared protocol for audit purposes. Coefficients and threshold are still selected on the same routing-calibration partition, which can make its calibration score optimistic without using future labels. Tables 5-13 retain the archived protocol; Section 7.4 reports the separate corrected-split validation, and neither replaces the other retrospectively.")

    p = next(p for p in doc.paragraphs if p.text.startswith("The audit tests controlled conditions and natural-year comparisons"))
    insert_after(p,
        "For the Yearbook injected-shift screen, all model arms are evaluated on the same later-year images before and after three fixed transformations: Gaussian blur (17-pixel kernel, sigma 3), a centered square occlusion covering one-third of image width and height, and additive Gaussian noise with standard deviation 0.6 in normalised pixel units. A seeded clean split-half comparison provides the null control. Feature MMD uses 499 paired within-image permutations and the domain classifier uses pair-grouped five-fold cross-validation. Because these perturbations are synthetic, their detectability does not establish their prevalence in natural Yearbook data.")

    # Update broad method claim in the system section.
    p = next(p for p in doc.paragraphs if p.text.startswith("The intended setting is covariate drift"))
    set_paragraph(p,
        "The intended setting is covariate shift, where class meaning remains stable while image acquisition or context changes. Candidate causes include camera replacement, illumination, weather, compression, sensor noise, and partial obstruction. The present evidence does not show that saliency solves these causes generally. In CLEAR, the learned gate contributes little under direct weight replacement. In Yearbook, saliency-ranked masking helps relative to random masking under one centered occlusion, but not under blur or noise. Camera traps provide a plausible application for partial visibility and changing vegetation; hospital images more often motivate tests of scanner, protocol, staining, and site changes. A domain-specific temporal dataset must verify which interference actually occurs and harms predictions before saliency is claimed to correct it.")

    # Insert the Yearbook results as a new section before the saliency/routing analysis.
    h6 = next(p for p in doc.paragraphs if p.text == "6. Saliency and routing analysis")
    yearbook_anchor = next(p for p in doc.paragraphs if p.text.startswith("At dataset level, the matched-head result is positive only on CIFAR-10.1"))
    heading = insert_after(yearbook_anchor, "5.6 Chronological Yearbook saliency-mask experiment", "Heading 2")
    natural_paragraph = insert_after(heading,
        "The natural temporal comparison did not show historical-model harm: the frozen head scored 87.62% macro F1 on the 1980-1989 selection sample and 93.13% on the 2004-2013 future sample. This later window is easier for this trained head, so chronological separation alone is not evidence of harmful drift. The injected-shift test therefore asks a different question: whether a known image change is measurable and reduces accuracy on those same future examples.")
    paragraph = insert_after(natural_paragraph,
        "Table 16 reports all three seeds for occlusion and seed 42 for blur and noise. Under center occlusion, saliency-mask LoRA beats the equal-area random-mask control by 0.30 to 1.49 macro-F1 points (mean 0.90) in all three seeds. It beats ordinary LoRA by 1.20 and 3.93 points in seeds 43 and 42, then trails ordinary LoRA by 0.29 points in seed 44. On clean images it trails ordinary LoRA by 0.90 to 1.50 points. Under blur it trails random masking in all seeds, and under noise its effect varies substantially. The same 335-image test set is used in all seeds, so seed variation does not create independent test samples. These results are exploratory, and the occlusion condition was examined before the three-seed repeat rather than preregistered as the primary endpoint.")
    yearbook_table = insert_table_before(doc, h6, [
        ["Seed", "Injected shift", "Historical accuracy drop", "Ordinary LoRA F1", "Random / saliency mask F1"],
        ["42", "Center occlusion", "7.16 points", "89.20%", "91.64% / 93.13%"],
        ["43", "Center occlusion", "4.78 points", "89.53%", "90.43% / 90.73%"],
        ["44", "Center occlusion", "5.08 points", "91.93%", "90.74% / 91.64%"],
        ["42", "Gaussian blur", "0.29 points", "96.12%", "94.63% / 94.33%"],
        ["42", "Gaussian noise", "8.35 points", "85.45%", "89.20% / 85.18%"],
    ])
    cap = insert_after(paragraph, "Table 16. Yearbook historical-model harm and future macro F1 under paired injected shifts (occlusion: three seeds; blur and noise: seed 42; same 335 future images)")
    cap._p.addnext(yearbook_table._tbl)
    insert_after(paragraph,
        "The clean split-half control is non-significant in all three seeds (MMD p = 0.272-0.994). Center occlusion produces MMD p = 0.002, domain-classifier AUC = 1.00, and a 4.78-7.16 point historical accuracy drop. Blur is highly detectable but causes little historical-model harm (accuracy change ranges from a 1.19-point improvement to a 1.20-point drop across seeds). Noise produces a 8.35-10.15 point drop and AUC = 1.00, yet saliency masking does not reliably correct it. Detectability and saliency benefit are therefore condition-specific. The test concerns controlled perturbations applied to later-year images, not natural Yearbook corruption or natural drift correction.")

    # Refresh key discussion claims while retaining the detailed historical CLEAR audit.
    p = next(p for p in doc.paragraphs if p.text.startswith("The three experiments support different levels of claim."))
    set_paragraph(p,
        "The evidence supports different claims at different strengths. CLEAR verifies several controlled feature shifts, but does not detect the sampled clean-year change. Its archived one-seed comparison favours saliency over conventional fine-tuning, while the corrected-split comparison reverses by 0.77 points and the learned-weight intervention finds little causal dependence on the gate. The Yearbook pilot detects injected occlusion, blur, and noise; only occlusion yields a repeated saliency advantage over random masking, and ordinary LoRA remains competitive. Natural Yearbook chronology produces no historical-model performance decline in the sampled windows. Routing remains a separate audit mechanism without established incremental selection value. These findings do not establish saliency correction of natural temporal drift.")
    p = next(p for p in doc.paragraphs if p.text.startswith("For RQ1, the original class-balanced 300-image"))
    set_paragraph(p,
        "For RQ1, the CLEAR random-sample audit detects motion blur and the evidence-ranked mixture at 100, 300, and 500 images per group, while the sampled clean 2009-to-2014 comparison remains non-significant. In Yearbook, injected occlusion, blur, and noise are all detectable in frozen features (paired MMD p = 0.002; domain AUC about 1.00), but the clean split-half control is not. Only occlusion causes a repeated, substantial historical-model drop alongside a saliency-mask advantage over random masking. The natural Yearbook historical model scores higher on the later sample than on the earlier selection sample. Drift detection, task harm, and saliency benefit therefore remain distinct.")
    p = next(p for p in doc.paragraphs if p.text.startswith("The source-separated results show the largest gains"))
    set_paragraph(p,
        "The current results do not support a general statement that saliency helps under other interferences. In the new Yearbook screen, saliency masking is below random masking under blur in all three seeds. Under noise it trails random masking by 4.02 and 5.84 points in two seeds and is 0.31 points higher in the third. This conflicts with the broad robustness interpretation one might draw from the occlusion result. The next test should use a real time-by-camera or time-by-acquisition dataset and should freeze a harmful shift before comparing saliency-guided adaptation with ordinary fine-tuning and matched controls.")
    p = next(p for p in doc.paragraphs if p.text.startswith("The main CLEAR evidence is limited by one random seed"))
    set_paragraph(p,
        "The CLEAR evidence uses one random seed, one temporal dataset, synthetic stressors layered over natural chronology, and one fitted policy. The archived CLEAR arms and corrected-split arms differ in the historical head and selection protocol. Yearbook adds three training seeds for injected perturbations, but all use the same 335 future images and the occlusion condition was explored before those repeats. Yearbook is a low-resolution binary portrait task; its sampled chronology does not show task harm. The planned iWildCam screen is not yet complete because the public 12 GB archive could not be retrieved quickly from the server. CIFAR-10.1 tests recollection, while Camelyon17 tests hospital shift rather than chronology. These limitations preclude a claim of general saliency benefit or natural drift correction.")

    # Replace future-work and conclusion with a plan that follows the evidence.
    p = next(p for p in doc.paragraphs if p.text.startswith("For RQ1, repeat the unlabelled random-sample"))
    set_paragraph(p,
        "The next experiment should use iWildCam-WILDS as a separate dataset and branch. First inspect the metadata for supported year-by-camera-by-species cells, then freeze a chronology that keeps sequence bursts together and reports both within-camera temporal change and held-out-camera performance. Measure clean split-half MMD, chronological MMD, domain AUC, and historical-model harm before any adaptation. If a harmful period exists, compare ordinary LoRA, saliency-guided adaptation, and a matched random spatial control across the same update IDs and seeds. Report day/night, illumination, blur, weather, and vegetation proxies only where metadata or visual audits support them. If the historical model has no chronological performance decline, do not move the test window after observing model scores; use the benchmark's native camera split as a separate acquisition-shift analysis.")
    p = next(p for p in doc.paragraphs if p.text.startswith("The project contributes a three-stage evaluation"))
    set_paragraph(p,
        "This project provides a controlled audit of saliency as a model-maintenance signal. The existing CLEAR architecture, weight-swap intervention, router comparisons, and three-seed Yearbook mask experiment show why saliency must be tested against ordinary updating and matched spatial controls. The strongest new positive finding is a small advantage over random masking under one injected Yearbook occlusion; the result is not consistent across blur, noise, or ordinary LoRA. Natural temporal harm has not been demonstrated in the current Yearbook split, and the corrected CLEAR run does not show a saliency-specific advantage. The contribution is therefore a carefully delimited feasibility study and an evaluation protocol, with iWildCam as the next independent test.")
    p = next(p for p in doc.paragraphs if p.text.startswith("The strongest positive result still concerns one CLEAR-100 seed"))
    set_paragraph(p,
        "The current evidence does not establish correction of natural temporal drift. CLEAR's archived saliency advantage reverses under the corrected split, and its gate intervention has near-zero effect. Yearbook shows a repeated 0.30-1.49 point advantage over a random-mask control under centered occlusion, but the test condition is synthetic and the mean advantage is only 0.90 points across three training seeds on one shared future sample. Under blur and noise the saliency method is not consistently competitive. The router also trails always-updated inference in the archived main comparison. These findings justify testing saliency as a targeted adaptation prior when a measured shift threatens class evidence, while keeping ordinary fine-tuning as the baseline.")
    p = next(p for p in doc.paragraphs if p.text.startswith("The next acceptance decision should rerun conventional"))
    set_paragraph(p,
        "The next decision is whether iWildCam's public metadata support a defensible time-by-camera experiment with measurable chronological harm. If they do, run MMD and domain-classifier checks before adaptation and evaluate the same later images under the native chronology. Compare saliency-mask adaptation with random masks and ordinary LoRA at matched IDs, parameter budgets, and seeds. If access remains too slow, record the blocker and proceed with the largest locally available time-indexed image benchmark. Keep hospital and camera datasets separate, and reserve the router for a later evaluation after the saliency intervention shows repeatable task value.")

    p = next(p for p in doc.paragraphs if p.text.startswith("RQ1 asked whether limited incoming images"))
    set_paragraph(p,
        "RQ1 asked whether limited incoming images reveal drift independently of task performance. The CLEAR unlabelled audit detects controlled motion blur and the evidence-ranked mixture at 100-500 images per group but does not detect the sampled clean-year difference. On Yearbook, paired MMD and domain AUC detect injected blur, occlusion, and noise, while the clean split-half control remains non-significant. Detection does not guarantee harm: Yearbook blur barely changes historical accuracy, while occlusion and noise reduce it by 4.78-7.16 and 8.35-10.15 points respectively. Natural chronological Yearbook performance does not decline in the sampled windows.")
    p = next(p for p in doc.paragraphs if p.text.startswith("RQ2 asked whether saliency-guided updating adds value"))
    set_paragraph(p,
        "RQ2 asked whether saliency adds value beyond ordinary fine-tuning and generic spatial regularisation. The CLEAR archived gain over conventional fine-tuning is not reproduced by the corrected split, and learned-weight interventions leave predictions almost unchanged. In the separate Yearbook adaptation, saliency masking exceeds equal-area random masking by 0.30-1.49 macro-F1 points under one injected occlusion in all three seeds (mean +0.90), but trails ordinary LoRA in one seed and loses under blur and usually noise. This is a narrow signal for one perturbation, not a general saliency-specific improvement.")

    # Update external-screen paragraphs that otherwise call Yearbook a counterexample without context.
    p = next(p for p in doc.paragraphs if p.text.startswith("At dataset level, the matched-head result is positive only on CIFAR-10.1"))
    set_paragraph(p,
        "At dataset level, the matched frozen-head result is positive only on CIFAR-10.1 (+0.30 ± 0.11 macro-F1 points across three seeds); Yearbook is -0.33 points in the original single-run head screen, and Camelyon17 is -0.07 ± 0.12 points. These screens do not show a universal benefit. The later Yearbook saliency-mask experiment is a separate method and split: it yields a small advantage over random masking only under injected occlusion. The two Yearbook results answer different questions and should not be collapsed into one estimate.")

    p = next(p for p in doc.paragraphs if p.text.startswith("Figure 4 shows the four strategy trajectories"))
    set_paragraph(p,
        "Figure 4 shows the archived four-strategy trajectories as the evidence-ranked mixture becomes more severe across the frozen horizon. The clean panel uses a narrow vertical scale because all four scores remain near 95%; the reported values therefore matter more than the apparent line slopes. In this single-seed archived comparison, the updated saliency-guided ViT produces the highest macro F1 in each future year. This advantage is not reproduced by the corrected-split comparison in Section 7.4, which uses a different historical head and selection partition.")

    doc.core_properties.title = "Saliency as an Operational Signal for Drift-Aware Vision Model Updating"
    doc.core_properties.subject = "Updated FYP report draft with CLEAR and Wild-Time Yearbook evidence"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
