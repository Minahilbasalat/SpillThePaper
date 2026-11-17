"""Markup, CSS and charts for the My results page (upload, mark, weak areas, history)."""

from __future__ import annotations

from datetime import datetime

import streamlit as st
from theme import DOT, esc

CSS = """
[class*="st-key-rs-card"],.st-key-rs-taped{position:relative;background:var(--cream);border:2px solid var(--ink);
border-radius:3px;box-shadow:6px 6px 0 var(--ink);padding:1.1rem 1.4rem 1.2rem;margin-bottom:1.3rem;gap:.6rem;}
.st-key-rs-taped{transform:rotate(-.6deg);margin-top:1rem;}
.st-key-rs-taped::before{content:"";position:absolute;top:-16px;left:34px;width:96px;height:28px;
background:rgba(255,213,66,.9);border:1px solid rgba(0,0,0,.12);transform:rotate(-3deg);}
.rs-h{display:flex;justify-content:space-between;align-items:flex-start;gap:.8rem;flex-wrap:wrap;}
.rs-t{font-family:'Archivo Black',sans-serif;font-size:1.2rem;line-height:1.2;color:var(--ink);}
.rs-sub{font-family:'Special Elite',monospace;font-size:.8rem;color:var(--ink);}
.rs-badges{display:flex;gap:.4rem;flex-wrap:wrap;align-items:center;}
.rs-badge{display:inline-block;border:1.5px solid var(--ink);padding:.05rem .5rem;font-family:'Special Elite',monospace;
font-size:.78rem;color:var(--ink);background:#fff;white-space:nowrap;}
.rs-badge.weak{background:var(--pink);}.rs-badge.ok{background:var(--yellow);}.rs-badge.strong{background:var(--blue);}
.rs-badge.dark{background:var(--ink);color:var(--lime);}
.rs-big{font-family:'Archivo Black',sans-serif;font-size:clamp(3rem,7vw,5rem);line-height:1;color:var(--ink);}
.rs-big small{font-size:.4em;}
.rs-text{font-family:'Special Elite',monospace;font-size:.95rem;line-height:1.55;color:var(--ink);}
.rs-ans{font-family:'Special Elite',monospace;font-size:.88rem;line-height:1.5;color:var(--ink);white-space:pre-wrap;
background:#fff;border:1.5px dashed var(--ink);padding:.6rem .8rem;margin:.2rem 0;}
.rs-list{margin:.2rem 0 .2rem 1.1rem;padding:0;font-family:'Special Elite',monospace;font-size:.88rem;color:var(--ink);}
.rs-table{width:100%;border-collapse:collapse;font-family:'Special Elite',monospace;font-size:.86rem;color:var(--ink);}
.rs-table th{text-align:left;border-bottom:2px solid var(--ink);padding:.3rem .4rem;}
.rs-table td{border-bottom:1px dashed var(--ink);padding:.3rem .4rem;vertical-align:top;}
.rs-note{font-size:.76rem;font-style:italic;color:var(--ink);}
.stApp [data-testid="stTextArea"] textarea,.stApp [data-testid="stNumberInput"] input{background:#fff !important;
border:2px solid var(--ink) !important;border-radius:2px !important;color:var(--ink) !important;
font-family:'Special Elite',monospace;}
.stApp [data-testid="stFileUploaderDropzone"]{background:#fff;border:2px dashed var(--ink);border-radius:2px;}
.stApp [data-testid="stExpander"] details{background:var(--cream);border:2px solid var(--ink);border-radius:2px;}
.stApp [class*="st-key-rs-go"] button{background:var(--ink) !important;box-shadow:5px 5px 0 var(--red) !important;}
.stApp [class*="st-key-rs-go"] button p{color:var(--lime) !important;}
.stApp [class*="st-key-rs-danger"] button{background:var(--pink) !important;}
"""

STATUS_TEXT = {
    "unanswered": "No answer found (counts 0)",
    "unreadable": "Could not be read (not counted)",
    "not_graded": "Not marked yet",
}
METHOD_TEXT = {
    "vision": "read by AI vision",
    "ocr": "read by OCR",
    "text": "typed text",
    "mixed": "read from several sources",
    "none": "nothing was read",
}
EXAM_LABELS = {"final": "Final", "mid1": "Mid 1", "mid2": "Mid 2", "quiz": "Quiz"}
MARKED = ("graded_ai", "manual", "unanswered")
BAND_COLORS = {"weak": "#FF9CC7", "ok": "#FFD542", "strong": "#8FD6FF"}


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- small helpers
def ratio_band(score: object, marks: object) -> str:
    """'weak' (< 50%), 'ok', 'strong' (75%+) or 'none' when not marked."""
    if score is None or not marks:
        return "none"
    pct = 100 * float(score) / float(marks)
    return "weak" if pct < 50 else "strong" if pct >= 75 else "ok"


def can_report(grades: list[dict] | None) -> bool:
    """True when at least one question carries a mark that counts."""
    return any(g.get("score") is not None and g.get("status") in MARKED for g in grades or [])


def date_text(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return str(iso)


def status_line(grade: dict) -> str:
    """Text under the question: the AI/manual feedback, or what the status means."""
    status = grade.get("status")
    if status in STATUS_TEXT:
        return STATUS_TEXT[status] if status != "not_graded" or not grade.get("feedback") else (
            f"{STATUS_TEXT[status]}. {grade['feedback']}")
    return grade.get("feedback") or ""


def exam_for(paper: dict) -> str | None:
    """Exam value for a personalised paper, or None for plain 'mid' / 'other'."""
    from state import exam_for_kind

    return exam_for_kind(str(paper.get("exam_kind") or ""))


def collect_files(uploaded: list) -> list[dict]:
    """Turn st.file_uploader files into the dictionaries read_answers() expects."""
    return [{"name": f.name, "data": f.getvalue()} for f in uploaded or []]


def combine_sheet(sheet, typed_answers: dict, typed_rest: str, numbers: list):
    """Add typed answers ('Q1 ... Q2 ...') to a sheet read from files. Returns the same sheet."""
    for n, text in typed_answers.items():
        sheet.answers[n] = (sheet.answers[n] + "\n" + text) if n in sheet.answers else text
    if typed_rest.strip():
        if len(numbers) == 1:
            sheet.answers[numbers[0]] = typed_rest.strip()
        else:
            sheet.unassigned = (sheet.unassigned + "\n" + typed_rest).strip()
            sheet.warnings.append("Some typed text could not be matched to a question. Start each answer "
                                  "with 'Q1', 'Q2' ... or copy it into the right box below.")
    if typed_answers or typed_rest.strip():
        sheet.method = "text" if sheet.method in ("none", "text") else "mixed"
    missing = [n for n in numbers if n not in sheet.answers]
    if sheet.answers and missing and not any(w.startswith("No answer found for question") for w in sheet.warnings):
        sheet.warnings.append("No answer found for question(s) " + ", ".join(map(str, missing)) +
                              " (they count as not answered unless you add them).")
    return sheet


# --------------------------------------------------------------------- markup
def grade_header(label: str, grade: dict) -> str:
    title = f"{label} {DOT} {grade['topic_label']}" if grade.get("topic_label") else label
    badges = []
    if grade.get("difficulty"):
        badges.append(f'<span class="rs-badge">{esc(str(grade["difficulty"]).capitalize())}</span>')
    score, marks = grade.get("score"), grade.get("marks")
    band = ratio_band(score, marks)
    shown = "not marked" if score is None else f"{float(score):g} / {float(marks):g}"
    badges.append(f'<span class="rs-badge {"" if band == "none" else band}">{esc(shown)}</span>')
    return (f'<div class="rs-h"><div class="rs-t">{esc(title)}</div>'
            f'<div class="rs-badges">{"".join(badges)}</div></div>')


def missing_markup(missing: list) -> str:
    if not missing:
        return ""
    items = "".join(f"<li>{esc(m)}</li>" for m in missing)
    return f'<div class="rs-note">Points the answer lacks:</div><ul class="rs-list">{items}</ul>'


def answer_markup(text: str) -> str:
    return f'<div class="rs-ans">{esc(text) if text else "(no answer)"}</div>'


def table_markup(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="rs-table"><tr>{head}</tr>{body}</table>'


def topic_rows(analysis: dict) -> list[list]:
    rows = []
    for t in analysis["topics"]:
        note = " (based on one question: a hint, not a fact)" if t["confidence"] == "low" else ""
        rows.append([t["topic_label"] + note, f'{t["score"]:g} / {t["marks"]:g}',
                     f'{t["percent"]:.0f}%', t["band"], t["questions"]])
    return rows


def group_rows(groups: dict) -> list[list]:
    return [[str(k), f'{v["score"]:g} / {v["marks"]:g}', f'{v["percent"]:.0f}%', v["questions"]]
            for k, v in sorted(groups.items(), key=lambda kv: kv[1]["percent"])]


def weakest_labels(attempt: dict, n: int = 3) -> list[str]:
    """Names of the weakest topics of a saved attempt (only those under 50%)."""
    topics = sorted(attempt.get("topics", []), key=lambda t: t["percent"])
    return [t["topic_label"] for t in topics if t["percent"] < 50][:n]


def attempt_markup(attempt: dict) -> str:
    pct = attempt["overall"]["percent"]
    weak = weakest_labels(attempt)
    badges = [f'<span class="rs-badge {ratio_band(pct, 100)}">{pct:.0f}%</span>']
    if attempt.get("personalised"):
        badges.append('<span class="rs-badge dark">personalised</span>')
    sub = f'{date_text(attempt["time"])} {DOT} {attempt["overall"]["score"]:g} / {attempt["overall"]["total"]:g}'
    weak_text = ("Weakest: " + "; ".join(weak)) if weak else "No weak area in this paper."
    return (f'<div class="rs-h"><div class="rs-t">{esc(attempt["title"])}</div>'
            f'<div class="rs-badges">{"".join(badges)}</div></div>'
            f'<div class="rs-sub">{esc(sub)}</div><div class="rs-text">{esc(weak_text)}</div>')


# --------------------------------------------------------------------- charts
def _layout(fig, height: int) -> None:
    fig.update_layout(height=height, margin=dict(l=0, r=10, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#FCF6E4", font=dict(family="Space Grotesk, sans-serif", color="#111111"),
                      showlegend=False)


def topic_figure(analysis: dict):
    """Horizontal bars: percent per topic, coloured by band (weak pink, ok yellow, strong blue)."""
    topics = analysis.get("topics") or []
    if not topics:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    rows = topics[::-1]
    fig = go.Figure()
    fig.add_bar(x=[t["percent"] for t in rows], y=[t["topic_label"] for t in rows], orientation="h",
                marker=dict(color=[BAND_COLORS[t["band"]] for t in rows], line=dict(color="#111111", width=1.5)))
    _layout(fig, max(240, 34 * len(rows) + 90))
    fig.update_xaxes(range=[0, 100], title="Share of the marks you earned (%)")
    return fig


def history_figure(attempts: list[dict]):
    """Line of the overall percent over the attempts of one subject (oldest first)."""
    if not attempts:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    rows = attempts[::-1]
    fig = go.Figure()
    fig.add_scatter(x=list(range(1, len(rows) + 1)), y=[r["overall"]["percent"] for r in rows],
                    mode="lines+markers", line=dict(color="#111111", width=2),
                    marker=dict(size=11, color="#FFD542", line=dict(color="#111111", width=1.5)),
                    text=[r["title"] for r in rows], hovertemplate="%{text}<br>%{y:.0f}%<extra></extra>")
    _layout(fig, 280)
    fig.update_yaxes(range=[0, 100], title="Score (%)")
    fig.update_xaxes(title="Attempt", dtick=1)
    return fig


def weakness_figure(weak: dict):
    """Cumulative weak areas of a subject (store.weakness), weakest first."""
    if not weak:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    rows = sorted(weak.values(), key=lambda v: -v["weakness"])[:10][::-1]
    fig = go.Figure()
    fig.add_bar(x=[r["percent"] for r in rows], y=[r["topic_label"] for r in rows], orientation="h",
                marker=dict(color=[BAND_COLORS["weak" if r["percent"] < 50 else "strong" if r["percent"] >= 75
                                   else "ok"] for r in rows], line=dict(color="#111111", width=1.5)))
    _layout(fig, max(240, 34 * len(rows) + 90))
    fig.update_xaxes(range=[0, 100], title="Marks earned, last attempts (%)")
    return fig
