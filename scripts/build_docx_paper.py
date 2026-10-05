"""Build an editable Word version of the completed transparent-router paper."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "transparent_router" / "long_horizon_mock_paper.md"
OUTPUT = ROOT / "output" / "docx" / "transparent_router_mock_paper.docx"

DARK_BLUE = "17324D"
PALE_BLUE = "EDF4F8"
LIGHT_BORDER = "D9D9D9"

SOURCE_REFERENCES = [
    "[6] Q. Yao and X. Gong, Saliency Guided Self-attention Network for Weakly and Semi-supervised Semantic Segmentation, arXiv:1910.05475, 2020.",
    "[7] H. Nguyen et al., XGD Explainable AI Guided Knowledge Distillation with Feature Refinement for Semantic Segmentation, 38th Canadian Conference on Artificial Intelligence, 2025.",
    "[8] Y. Cheng, X. Liu, S. Chen, and C. Xu, Siamese ViT A Local Global Feature Fusion Method for Real Time Visual Navigation of UAVs in Real World Environments, Remote Sensing, vol. 18, article 1556, 2026, doi:10.3390/rs18101556.",
    "[9] R. Cantini, A. Orsino, and D. Talia, XAI driven knowledge distillation of large language models for efficient deployment on low resource devices, Journal of Big Data, vol. 11, article 63, 2024, doi:10.1186/s40537-024-00928-3.",
    "[10] S. Zheng, A. Ghosh, and S. G. Narasimhan, Saliency Guided Image Warping for Unsupervised Domain Adaptation, arXiv:2403.12712, 2024.",
    "[11] A. A. Ismail, S. Feizi, and H. Corrada Bravo, Improving Deep Learning Interpretability by Saliency Guided Training, Advances in Neural Information Processing Systems 35, 2021.",
    "[12] K. Yang et al., Semantic Change Detection with Asymmetric Siamese Networks, arXiv:2010.05687, 2021.",
]


def set_cell_fill(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    properties = table._tbl.tblPr
    borders = properties.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), LIGHT_BORDER)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instruction, end))


def add_inline(paragraph, text: str) -> None:
    parts = re.split(r"(\*\*.*?\*\*|`.*?`|\*.*?\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif part.startswith("*") and part.endswith("*"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        else:
            paragraph.add_run(part)


def add_table(document: Document, lines: list[str]) -> None:
    parsed = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [parsed[0], *parsed[2:]]
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            cell = table.cell(row_index, column_index)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_index == 0:
                set_cell_fill(cell, DARK_BLUE)
            elif row_index % 2 == 0:
                set_cell_fill(cell, PALE_BLUE)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if column_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            add_inline(paragraph, value)
            for run in paragraph.runs:
                run.font.size = Pt(8.5)
                if row_index == 0:
                    run.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def add_related_work(document: Document) -> None:
    document.add_paragraph("2.3 Related work and design rationale", style="Heading 2")
    paragraphs = [
        "The supplied literature supports treating explanations as part of learning rather than as detached illustrations. SGAN constrains self-attention with saliency priors to reduce attention spreading into irrelevant background regions [6]. XGD combines prediction alignment with saliency-based feature refinement and validates the contribution of its losses through ablation [7]. Ismail et al. mask low-saliency features and enforce output consistency, showing how perturbation and predictive faithfulness can be optimized together [11]. These findings motivate the prototype's learned patch gate, complementary low-saliency path, and explicit auxiliary saliency losses.",
        "The paired-model design follows the comparison logic of Siamese systems while preserving temporal asymmetry. Siamese-ViT demonstrates that weight-sharing ViT branches can form a comparable representation for cross-view matching [8]. Asymmetric Siamese Networks show that paired observations from different distributions may require deliberately different processing roles [12]. Accordingly, this project initializes both branches from the same historical XAI ViT, freezes the old branch as an auditable reference, and adapts only the newer branch to the 2010 drift stream.",
        "The drift rationale also follows work on domain shift. Saliency Guided Image Warping reports that changing backgrounds can dominate adaptation while smaller foreground objects remain underrepresented [10]. DiXtill shows that explanation agreement can be transferred alongside predictive behavior rather than relying on output logits alone [9]. The current prototype therefore logs saliency divergence together with confidence and entropy changes, and requires visible evidence change before an updated prediction is eligible. It does not claim object-level localization because CLEAR-100 provides class labels rather than segmentation masks.",
    ]
    for text in paragraphs:
        paragraph = document.add_paragraph()
        add_inline(paragraph, text)

    rows = [
        ["Evidence from supplied paper", "Design choice in this project", "Limit retained in evaluation"],
        ["Saliency priors constrain attention spread [6]", "Learned patch saliency directly weights the ViT representation", "Patch gates are not object masks"],
        ["Prediction and saliency losses can be ablated separately [7]", "Matched run disables only the two auxiliary saliency losses", "One seed cannot establish a causal effect"],
        ["Shared ViT branches support comparable representations [8]", "Historical and newer branches share the same ViT architecture and initialization", "Only the newer branch is adapted"],
        ["Explanation agreement can guide transfer [9]", "Old and new saliency distributions enter the adaptation and routing logic", "The router remains a calibrated correlation model"],
        ["Background variation can dominate domain adaptation [10]", "Realistic noise, saturation, and occlusion stress contextual dependence", "Synthetic corruption is reported separately from temporal drift"],
        ["Masking and output consistency improve saliency faithfulness [11]", "Complementary low-saliency evidence is penalized during training", "Faithfulness still needs deletion and insertion evaluation"],
        ["Temporal pairs may require asymmetric roles [12]", "Frozen historical reference is compared with an adapted newer ViT", "2011 selection and calibration reuse remains a limitation"],
    ]
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    lines.insert(1, "|---|---|---|")
    add_table(document, lines)


def build() -> None:
    document = Document()
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    title = styles["Title"]
    title.font.name = "Aptos Display"
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title.paragraph_format.space_after = Pt(8)
    title_properties = title.element.get_or_add_pPr()
    title_border = title_properties.find(qn("w:pBdr"))
    if title_border is not None:
        title_properties.remove(title_border)

    for name, size, before, after in (("Heading 1", 16, 14, 7), ("Heading 2", 13, 11, 5), ("Heading 3", 11.5, 9, 4)):
        style = styles[name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.LEFT
    footer.paragraph_format.tab_stops.add_tab_stop(Inches(6.75), WD_TAB_ALIGNMENT.RIGHT)
    footer.add_run("Rik FYP transparent router prototype")
    footer.runs[0].font.size = Pt(8)
    footer.runs[0].font.color.rgb = RGBColor(95, 105, 120)
    footer.add_run("\t")
    add_page_number(footer)

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = document.add_paragraph()
        add_inline(paragraph, " ".join(line.strip() for line in paragraph_lines))
        paragraph_lines.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped == "[PAGEBREAK]":
            flush_paragraph()
            document.add_page_break()
            index += 1
            continue
        if stripped.startswith("!["):
            flush_paragraph()
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if match:
                caption, relative_path = match.groups()
                image_path = SOURCE.parent / relative_path
                caption_paragraph = document.add_paragraph()
                caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                caption_run = caption_paragraph.add_run(caption)
                caption_run.bold = True
                caption_run.font.size = Pt(9)
                caption_paragraph.paragraph_format.keep_with_next = True
                picture_paragraph = document.add_paragraph()
                picture_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                picture_paragraph.add_run().add_picture(str(image_path), width=Inches(6.65))
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[index + 1].strip()):
            flush_paragraph()
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            add_table(document, table_lines)
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = heading.group(2)
            if text == "1. Research question":
                keyword_paragraph = document.add_paragraph()
                keyword_run = keyword_paragraph.add_run("Keywords  ")
                keyword_run.bold = True
                keyword_paragraph.add_run("temporal data drift, vision transformer, explainable artificial intelligence, saliency, Siamese network, transparent routing")
                text = "1. Introduction and research question"
            style = "Title" if level == 1 else f"Heading {level - 1}"
            paragraph = document.add_paragraph(style=style)
            add_inline(paragraph, text)
            if level == 1:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                properties = paragraph._p.get_or_add_pPr()
                border = properties.find(qn("w:pBdr"))
                if border is not None:
                    properties.remove(border)
            index += 1
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            paragraph = document.add_paragraph(style="List Bullet")
            add_inline(paragraph, stripped[2:])
            index += 1
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.25)
            paragraph.paragraph_format.first_line_indent = Inches(-0.2)
            paragraph.add_run(f"{match.group(1)}.  ")
            add_inline(paragraph, match.group(2))
            index += 1
            continue
        if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            flush_paragraph()
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(stripped.strip("*"))
            run.italic = True
            run.font.color.rgb = RGBColor(70, 80, 90)
            index += 1
            continue
        paragraph_lines.append(line)
        index += 1
    flush_paragraph()

    document.core_properties.title = "Transparent Saliency Routing for Temporal Image Drift"
    document.core_properties.subject = "Completed fixed-horizon prototype report"
    document.core_properties.author = "Rik FYP"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
