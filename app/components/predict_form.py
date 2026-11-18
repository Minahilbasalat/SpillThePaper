"""CSS and small helpers for the Upcoming paper form: cards, chips, exam buttons, difficulty bar."""

from __future__ import annotations

import streamlit as st
from theme import esc

EXAMS = [("final", "Final"), ("mid1", "Mid 1"), ("mid2", "Mid 2"), ("quiz", "Quiz")]
DURATIONS = [(30, "30 min"), (60, "1 hr"), (90, "90 min"), (180, "3 hrs")]

CSS = """
[class*="st-key-card-"]{background:var(--cream);border:2px solid var(--ink);border-radius:3px;
box-shadow:6px 6px 0 var(--ink);padding:1.2rem 1.4rem 1.4rem;margin-bottom:1.1rem;gap:.8rem;}
.card-h{display:flex;align-items:center;gap:.7rem;font-family:'Archivo Black',sans-serif;font-size:1.25rem;color:var(--ink);}
.num{width:2rem;height:2rem;background:var(--ink);color:var(--lime);display:inline-flex;align-items:center;
justify-content:center;font-size:1rem;}
.opt{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:.6rem;background:var(--blue);
border:1.5px solid var(--ink);padding:.1rem .4rem;letter-spacing:.04em;}
.card-sub{font-family:'Special Elite',monospace;font-size:.85rem;color:var(--ink);}

[class*="st-key-chip-"] [data-testid="stButton"]>button,[class*="st-key-pick-"] [data-testid="stButton"]>button{
border-radius:2px;box-shadow:none;padding:.4rem .6rem;width:100%;white-space:nowrap;}
[class*="st-key-chip-"] [data-testid="stButton"]>button p,[class*="st-key-pick-"] [data-testid="stButton"]>button p{font-size:.8rem;}
[class*="st-key-chip-on"] [data-testid="stButton"]>button{background:var(--ink) !important;}
[class*="st-key-chip-on"] [data-testid="stButton"]>button p{color:var(--lime) !important;}
[class*="st-key-pick-on"] [data-testid="stButton"]>button{background:var(--yellow) !important;box-shadow:3px 3px 0 var(--ink) !important;}

[class*="st-key-exam-"] [data-testid="stButton"]>button{width:100%;padding:.9rem 0;border-radius:3px;}
[class*="st-key-exam-"] [data-testid="stButton"]>button p{font-family:'Archivo Black',sans-serif !important;font-size:1.15rem;}
[class*="st-key-exam-on"] [data-testid="stButton"]>button{background:var(--yellow) !important;box-shadow:5px 5px 0 var(--ink) !important;}

.st-key-up_more_sel [data-baseweb="select"]>div{background:var(--cream) !important;border:2px solid var(--ink) !important;
border-radius:2px !important;}
.st-key-up_custom_min input,.st-key-up_nq input,.st-key-up_marks input{background:#fff !important;}
[data-testid="stExpander"]{background:var(--cream);border:2px solid var(--ink) !important;border-radius:3px;}

.dbar{display:flex;height:18px;border:2px solid var(--ink);background:#fff;margin:.5rem 0 .3rem;}
.dbar span{display:block;height:100%;}
.dlabels{display:flex;justify-content:space-between;font-size:.72rem;font-weight:500;color:var(--ink);}
.dnote{font-family:'Special Elite',monospace;font-size:.68rem;color:var(--ink);margin-top:.5rem;}

.st-key-cta-gen{margin-top:1.1rem;}
.st-key-cta-gen [data-testid="stButton"]>button{background:var(--ink) !important;border:2px solid var(--ink);
box-shadow:6px 6px 0 var(--red) !important;width:100%;padding:1rem;border-radius:3px;}
.st-key-cta-gen [data-testid="stButton"]>button p{color:var(--lime) !important;font-weight:700;font-size:1rem;margin:0;}
.fine{font-family:'Special Elite',monospace;font-size:.72rem;color:var(--ink);margin:.8rem 0;}
[class*="st-key-chip-"] [data-testid="stElementContainer"],
[class*="st-key-pick-"] [data-testid="stElementContainer"],
[class*="st-key-exam-"] [data-testid="stElementContainer"],
[class*="st-key-chip-"] [data-testid="stButton"],
[class*="st-key-pick-"] [data-testid="stButton"],
[class*="st-key-exam-"] [data-testid="stButton"]{width:100% !important;}
[class*="st-key-exam-"] [data-testid="stButton"]>button{padding:.9rem .5rem !important;}
[data-testid="stExpander"] summary{background:var(--cream) !important;}
[data-testid="stExpander"] summary:hover{background:var(--yellow) !important;}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def card_heading(number: int, text: str, optional: bool = False) -> str:
    badge = '<span class="opt">OPTIONAL</span>' if optional else ""
    return f'<div class="card-h"><span class="num">{number}</span><span>{esc(text)}</span>{badge}</div>'


def chip_widths(labels: list[str]) -> list[float]:
    """Column weights proportional to the text length, plus one spare column on the right."""
    weights = [len(label) + 5 for label in labels]
    return weights + [max(1, 60 - sum(weights))]


def difficulty_percentages(mix: dict | None) -> list[int] | None:
    """[easy, medium, hard] as whole percentages, or None when the backend has no mix."""
    if not mix:
        return None
    values = [float(mix.get(key, 0) or 0) for key in ("easy", "medium", "hard")]
    total = sum(values)
    if total <= 0:
        return None
    pct = [round(v * 100 / total) for v in values]
    pct[1] = max(0, pct[1] + 100 - sum(pct))
    return pct


def difficulty_markup(pct: list[int]) -> str:
    colors = ["var(--blue)", "var(--cream)", "var(--pink)"]
    bar = "".join(f'<span style="width:{p}%;background:{c}"></span>' for p, c in zip(pct, colors))
    return (
        f'<div class="dbar">{bar}</div>'
        f'<div class="dlabels"><span>Easy {pct[0]}%</span><span>Medium {pct[1]}%</span><span>Hard {pct[2]}%</span></div>'
    )