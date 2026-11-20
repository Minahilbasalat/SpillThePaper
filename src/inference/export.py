"""Export a paper (and optionally its solutions) as a PDF. Pure Python (fpdf2).
The built-in PDF fonts only cover Latin-1, so other symbols are replaced by '?'."""
from __future__ import annotations

from fpdf import FPDF

from src.inference.layout import PART_LINE, layout_blocks, restore_parts


def _t(s) -> str:
    s = str(s if s is not None else "")
    for old, new in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u2013", "-"), ("\u2014", "-"), ("\u2022", "-"), ("\u2026", "...")):
        s = s.replace(old, new)
    return s.encode("latin-1", "replace").decode("latin-1")


def _question_text(pdf: FPDF, text: str) -> None:
    """Instruction and sub-parts on their own lines, code in Courier with its indentation."""
    for kind, body in layout_blocks(restore_parts(text)):
        if kind == "code":
            pdf.set_font("Courier", "", 9)
            pdf.set_fill_color(240, 240, 240)
            pdf.multi_cell(0, 4.5, _t(body), new_x="LMARGIN", new_y="NEXT", fill=True)
            pdf.ln(1)
            continue
        pdf.set_font("Helvetica", "", 10)
        for line in body.split("\n"):
            if line.strip():
                pdf.set_x(pdf.l_margin + (6 if PART_LINE.match(line.strip()) else 0))
                pdf.multi_cell(0, 5, _t(line.strip()), new_x="LMARGIN", new_y="NEXT")


def paper_to_pdf(paper: dict, solutions: list[dict] | None = None) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, _t(paper["title"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    meta = f"Time allowed: {paper['duration_minutes']} min"
    if paper.get("total_marks"):
        meta += f"   Total marks: {paper['total_marks']:.0f}"
    pdf.cell(0, 6, _t(meta), new_x="LMARGIN", new_y="NEXT")
    if paper.get("source") == "predicted":
        pdf.set_font("Helvetica", "I", 9)
        what = ("Tentative paper with questions written by AI from real past questions"
                if paper.get("question_style") == "new" else "Tentative paper built from real past questions")
        pdf.multi_cell(0, 5, what + ". It is a study aid, not a guarantee of what will appear.",
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    sol = {s["number"]: s for s in (solutions or [])}
    for q in paper["questions"]:
        pdf.set_font("Helvetica", "B", 11)
        marks = f"  [{q['marks']:.0f} marks]" if q.get("marks") else ""
        pdf.cell(0, 7, _t(f"Q{q['number']}{marks}"), new_x="LMARGIN", new_y="NEXT")
        _question_text(pdf, q["text"])
        s = sol.get(q["number"])
        if s and s["text"]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, _t(f"Solution ({s['source']})"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 4.5, _t(s["text"]), new_x="LMARGIN", new_y="NEXT")
            if s.get("marking_scheme"):
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(0, 4.5, _t("Marking: " + s["marking_scheme"]), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
    return bytes(pdf.output())
