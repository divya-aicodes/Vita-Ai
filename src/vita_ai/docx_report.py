"""Generate polished Vita AI screening reports as Word documents."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from .domain import ScreeningResult

BLUE = "1F4E78"
PALE_BLUE = "EAF2F8"
LIGHT_GRAY = "D9D9D9"
TEXT_GRAY = RGBColor(70, 70, 70)


def _set_cell_fill(cell, color: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), color)


def _set_cell_margins(
    cell, *, top: int = 80, start: int = 120, bottom: int = 80, end: int = 120
) -> None:
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


def _set_table_borders(table) -> None:
    properties = table._tbl.tblPr
    borders = properties.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = borders.find(qn(f"w:{edge}"))
        if border is None:
            border = OxmlElement(f"w:{edge}")
            borders.append(border)
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "6")
        border.set(qn("w:color"), LIGHT_GRAY)


def _repeat_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    properties.append(header)


def _set_cell_text(cell, text: str, *, bold: bool = False, color: RGBColor | None = None) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Arial"
    run.font.size = Pt(9.5)
    if color:
        run.font.color.rgb = color
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    _set_cell_margins(cell)


def _add_bullet(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.add_run(text)


def _remove_paragraph_border(paragraph_or_style) -> None:
    properties = paragraph_or_style._element.get_or_add_pPr()
    borders = properties.find(qn("w:pBdr"))
    if borders is not None:
        properties.remove(borders)


def build_docx_report(
    *,
    case_id: str,
    crop: str,
    symptoms: list[str],
    environments: list[str],
    distribution: str,
    notes: str,
    result: ScreeningResult,
) -> bytes:
    """Return an editable Word report as DOCX bytes."""
    del symptoms, environments, distribution
    document = Document()
    section = document.sections[0]
    section.start_type = WD_SECTION.NEW_PAGE
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.0
    for style_name, size in (("Title", 20), ("Heading 1", 13), ("Heading 2", 11)):
        style = styles[style_name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True
        _remove_paragraph_border(style)
        style.paragraph_format.space_before = Pt(8 if style_name != "Title" else 0)
        style.paragraph_format.space_after = Pt(3)

    properties = document.core_properties
    properties.title = "Vita AI Plant Health Screening Report"
    properties.subject = "Image based plant health screening result"
    properties.author = "Vita AI"

    title = document.add_paragraph(style="Title")
    _remove_paragraph_border(title)
    title.paragraph_format.space_after = Pt(2)
    title.add_run("Vita AI Plant Health Screening Report")
    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(9)
    subtitle_run = subtitle.add_run("Image based decision support from a selected ML classifier")
    subtitle_run.italic = True
    subtitle_run.font.color.rgb = TEXT_GRAY

    primary = result.primary
    confidence = primary.match_score if primary else None
    opening = document.add_paragraph()
    opening.paragraph_format.space_after = Pt(7)
    opening.add_run("Result summary. ").bold = True
    if primary:
        opening.add_run(
            f"The selected model ranked {primary.crop} - {primary.name} first at "
            f"{confidence:.1%} confidence. "
        )
    opening.add_run(
        "This result describes a controlled PlantVillage image class and should be confirmed "
        "before treatment, disposal, or another high impact decision."
    )

    document.add_heading("Case summary", level=1)
    generated = datetime.now(UTC).replace(microsecond=0).isoformat()
    summary_rows = (
        ("Case ID", case_id),
        ("Generated", generated),
        ("Screening method", "ML image classification" if result.method == "ml" else "Rule based"),
        ("Crop", crop),
        ("Outcome", result.headline),
        ("Photo quality", result.photo.status.title() if result.photo else "Not assessed"),
    )
    summary = document.add_table(rows=0, cols=2)
    summary.alignment = WD_TABLE_ALIGNMENT.LEFT
    summary.autofit = False
    summary.columns[0].width = Inches(1.65)
    summary.columns[1].width = Inches(5.2)
    _set_table_borders(summary)
    for label, value in summary_rows:
        cells = summary.add_row().cells
        _set_cell_text(cells[0], label, bold=True)
        _set_cell_fill(cells[0], "F2F2F2")
        _set_cell_text(cells[1], value)

    document.add_heading("Model assessment", level=1)
    document.add_paragraph(result.explanation)

    document.add_heading("Prediction shortlist", level=2)
    table = document.add_table(rows=1, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    widths = (Inches(0.55), Inches(1.25), Inches(2.45), Inches(1.25), Inches(1.3))
    headers = ("Rank", "Crop", "Condition", "Confidence", "Type")
    for index, (cell, header, width) in enumerate(
        zip(table.rows[0].cells, headers, widths, strict=True)
    ):
        cell.width = width
        _set_cell_fill(cell, BLUE)
        _set_cell_text(cell, header, bold=True, color=RGBColor(255, 255, 255))
        cell.paragraphs[0].alignment = (
            WD_ALIGN_PARAGRAPH.CENTER if index in {0, 3} else WD_ALIGN_PARAGRAPH.LEFT
        )
    _repeat_header(table.rows[0])
    _set_table_borders(table)
    for rank, candidate in enumerate(result.candidates, start=1):
        cells = table.add_row().cells
        values = (
            str(rank),
            candidate.crop,
            candidate.name,
            f"{candidate.match_score:.2%}",
            candidate.category.title(),
        )
        for index, (cell, value, width) in enumerate(zip(cells, values, widths, strict=True)):
            cell.width = width
            _set_cell_text(cell, value)
            if rank % 2 == 0:
                _set_cell_fill(cell, PALE_BLUE)
            if index in {0, 3}:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    if primary:
        document.add_heading("Recommended checks", level=1)
        for step in primary.next_steps:
            _add_bullet(document, step)
        document.add_heading("Prevention guidance", level=1)
        for step in primary.prevention:
            _add_bullet(document, step)

    if notes.strip() and not notes.strip().casefold().startswith("ml model:"):
        document.add_heading("Screening metadata", level=2)
        document.add_paragraph(notes.strip())

    document.add_heading("Limitations and safety", level=1)
    document.add_paragraph(
        "Model confidence is not field accuracy and does not prove a diagnosis. Real field images "
        "may differ from the controlled training dataset because of lighting, background, crop "
        "variety, growth stage, mixed conditions, pests, nutrition, or weather damage. Do not "
        "apply a pesticide, discard a crop, or make another high impact decision from this report "
        "alone. Seek local expert confirmation when symptoms spread, plants wilt rapidly, or the "
        "result conflicts with observations across several leaves."
    )

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer.add_run(
        "Vita AI | Decision support only | Uploaded images are not retained"
    )
    footer_run.font.name = "Arial"
    footer_run.font.size = Pt(8)
    footer_run.font.color.rgb = TEXT_GRAY

    output = BytesIO()
    document.save(output)
    return output.getvalue()
