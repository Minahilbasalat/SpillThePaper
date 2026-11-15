"""Export a paper (and optionally its solutions) as a PDF. Pure Python (fpdf2).
The built-in PDF fonts only cover Latin-1, so other symbols are replaced by '?'."""
from __future__ import annotations

from fpdf import FPDF


def _t(s) -> str:
    return str(s if s is not None else "").replace("\u2019", "'").replace("\u2013", "-") \
        .encode("latin-1", "replace").decode("latin-1")


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
        pdf.multi_cell(0, 5, "Tentative paper built from real past questions. It is a study aid, "
                             "not a guarantee of what will appear.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    sol = {s["number"]: s for s in (solutions or [])}
    for q in paper["questions"]:
        pdf.set_font("Helvetica", "B", 11)
        marks = f"  [{q['marks']:.0f} marks]" if q.get("marks") else ""
        pdf.cell(0, 7, _t(f"Q{q['number']}{marks}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _t(q["text"]), new_x="LMARGIN", new_y="NEXT")
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
