"""Markup, CSS and chart for the predicted paper screen (Upcoming paper, mockup page 5)."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from theme import DOT, esc

CSS = """
.st-key-paper-hero{position:relative;background:var(--cream);border:2px solid var(--ink);border-radius:3px;
box-shadow:8px 8px 0 var(--ink);transform:rotate(-.6deg);padding:2rem 2.4rem 1.6rem;margin-top:.9rem;gap:.2rem;}
.st-key-paper-hero::before{content:"";position:absolute;top:-16px;left:34px;width:96px;height:28px;
background:rgba(255,213,66,.9);border:1px solid rgba(0,0,0,.12);transform:rotate(-3deg);}
.ph-top{display:flex;justify-content:space-between;align-items:center;gap:1rem;}
.ph-h{font-family:'Archivo Black',sans-serif;font-size:clamp(2rem,4.2vw,3.2rem);line-height:1.05;color:var(--ink);margin:.6rem 0 .4rem;}
.ph-meta{font-family:'Special Elite',monospace;font-size:.9rem;color:var(--ink);}
.dash{border-top:2px dashed var(--ink);margin:.9rem 0 .5rem;}
[class*="st-key-pq-"]{padding:.7rem 0;gap:.3rem;}
.qn{font-family:'Archivo Black',sans-serif;font-size:2rem;line-height:1.1;color:var(--ink);}
.qt{font-family:'Special Elite',monospace;font-size:1.02rem;line-height:1.5;color:var(--ink);}
.qchips{display:flex;flex-wrap:wrap;gap:.45rem;align-items:center;margin-top:.55rem;}
.qchip{display:inline-block;border:1.5px solid var(--ink);padding:.05rem .45rem;font-family:'Special Elite',monospace;
font-size:.72rem;color:var(--ink);}
.qchip.blue{background:var(--blue);}
.qchip.warn{background:var(--yellow);}
.likely{font-family:'Special Elite',monospace;font-size:.72rem;letter-spacing:.06em;color:var(--ink);}
.tag-explore{display:inline-block;border:1.5px solid var(--red);color:var(--red);font-size:.72rem;padding:.1rem .5rem;white-space:nowrap;}
.stApp [class*="st-key-declass-"] [data-testid="stButton"]>button{background:var(--ink) !important;box-shadow:none !important;
border-radius:2px;padding:.45rem .9rem;}
.stApp [class*="st-key-declass-"] [data-testid="stButton"]>button p{color:#fff !important;font-size:.8rem;}

.st-key-next-move{background:#fff;border:2px solid var(--ink);box-shadow:6px 6px 0 var(--ink);padding:1.2rem 1.2rem 1.4rem;
gap:.8rem;margin-top:1.4rem;}
.nm-h{font-family:'Special Elite',monospace;font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;color:var(--ink);}
[class*="st-key-mv-"],[class*="st-key-mv-"] [data-testid="stElementContainer"],
[class*="st-key-mv-"] [data-testid="stButton"],[class*="st-key-mv-"] [data-testid="stDownloadButton"]{width:100% !important;}
.stApp [class*="st-key-mv-"] button{width:100%;border-radius:2px;padding:.8rem 1rem;box-shadow:none;}
.stApp [class*="st-key-mv-"] button p{font-weight:700;font-size:.95rem;}
.stApp .st-key-mv-start button{background:var(--ink) !important;box-shadow:5px 5px 0 var(--red) !important;}
.stApp .st-key-mv-start button p{color:var(--lime) !important;}
.stApp .st-key-mv-solution button{background:var(--pink) !important;}
.stApp .st-key-mv-upload button{background:var(--blue) !important;opacity:.55;}
.stApp .st-key-mv-dl button{background:transparent !important;border:0 !important;box-shadow:none !important;
text-decoration:underline;padding:.2rem;}
.soon{font-family:'Special Elite',monospace;font-size:.7rem;color:var(--ink);margin-top:-.4rem;}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def _blank(value: object) -> bool:
    return value is None or bool(pd.isna(value))


def hhmmss(seconds: int) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def duration_text(minutes: object) -> str:
    """'3 hrs', '1 hr', '90 min'."""
    if _blank(minutes):
        return "not set"
    m = int(minutes)
    if m % 60 == 0:
        hours = m // 60
        return f"{hours} hr" + ("" if hours == 1 else "s")
    return f"{m} min"


def squares(level: object) -> str:
    n = 0 if _blank(level) else max(0, min(5, int(level)))
    return "\u25a0" * n + "\u25a1" * (5 - n)


def question_label(question: dict, index: int) -> str:
    number = question.get("number")
    text = str(index + 1) if _blank(number) else str(number)
    return text if text.upper().startswith("Q") else f"Q{text}"


def header_markup(subject: str, exam_label: str, minutes: object, total_marks: object) -> str:
    kicker = f"Tentative paper {DOT} {subject} {DOT} {exam_label}"
    meta = [f"Time allowed {duration_text(minutes)}"]
    if not _blank(total_marks):
        meta.append(f"Total marks {float(total_marks):g}")
    return (
        f'<div class="ph-top"><div class="kicker">{esc(kicker)}</div><div class="stamp">LEAKED</div></div>'
        '<div class="ph-h">What your paper probably looks like</div>'
        f'<div class="ph-meta">{esc(f" {DOT} ".join(meta))}</div><div class="dash"></div>'
    )


def question_markup(question: dict) -> str:
    chips: list[str] = []
    if not _blank(question.get("marks")):
        chips.append(f'<span class="qchip">{float(question["marks"]):g} marks</span>')
    if question.get("marks_note"):
        chips.append('<span class="qchip warn">marks estimated</span>')
    if question.get("difficulty"):
        chips.append(f'<span class="qchip">{esc(str(question["difficulty"]).capitalize())}</span>')
    if not _blank(question.get("likelihood")):
        chips.append(f'<span class="likely">Likely {squares(question["likelihood"])}</span>')
    if question.get("leetcode_style"):
        chips.append('<span class="qchip blue">LeetCode-style</span>')
    if question.get("explore"):
        chips.append('<span class="tag-explore">Less-predicted area</span>')
    text = esc(question.get("text", "")).replace("\n", "<br>")
    return f'<div class="qt">{text}</div><div class="qchips">{"".join(chips)}</div>'


def redacted_markup() -> str:
    return (
        '<div class="redact" style="width:100%"></div>'
        '<div class="redact" style="width:62%"></div>'
    )


def realised_text(paper: dict) -> str:
    """Sentence for the yellow Difficulty note, written from paper['realised']."""
    realised = paper.get("realised") or {}
    parts = []
    for key, noun in (
        ("coding_questions", "coding question"),
        ("problem_solving_questions", "problem-solving question"),
        ("explore_questions", "surprise question"),
    ):
        n = int(realised.get(key, 0) or 0)
        if n:
            parts.append(f"{n} {noun}" + ("" if n == 1 else "s"))
    return (", ".join(parts) + ".") if parts else "A short paper: read the notes below to see why."


def topic_figure(topics: object):
    """Horizontal bar chart of topic probabilities, or None when plotly or data is missing."""
    if not isinstance(topics, pd.DataFrame) or topics.empty:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    top = topics.sort_values("probability", ascending=False).head(12).iloc[::-1]
    colors = {"Likely": "#FFD542", "Possible": "#8FD6FF", "Unlikely": "#D9D9CF"}
    fig = go.Figure()
    fig.add_bar(
        x=top["probability"],
        y=top["topic_label"],
        orientation="h",
        marker=dict(color=[colors.get(b, "#D9D9CF") for b in top["band"]], line=dict(color="#111111", width=1.5)),
    )
    if "examined_this_year" in top.columns:
        mask = top["examined_this_year"].fillna(False).astype(bool)
        if mask.any():
            fig.add_scatter(
                x=top.loc[mask, "probability"],
                y=top.loc[mask, "topic_label"],
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="#D4281A", line=dict(color="#111111", width=1)),
            )
    fig.update_layout(
        height=max(260, 30 * len(top) + 90),
        margin=dict(l=0, r=10, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FCF6E4",
        font=dict(family="Space Grotesk, sans-serif", color="#111111"),
        xaxis=dict(title="Relative likelihood (a ranking, not an exact chance)", rangemode="tozero"),
        showlegend=False,
    )
    return fig