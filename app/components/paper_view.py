"""Markup and CSS for the Past papers subject page (exam kinds and paper rows)."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from theme import DOT, esc

FILE_TYPES = {"pdf": "PDF", "doc": "DOC", "docx": "DOC", "jpg": "JPG", "jpeg": "JPG", "png": "PNG"}

CSS = """
.crumb{font-family:'Special Elite',monospace;font-size:.8rem;color:var(--ink);margin-top:.5rem;}
.crumb u{text-decoration:underline;}

[class*="st-key-kind-"]{position:relative;border:2px solid var(--ink);border-radius:3px;background:var(--cream);
box-shadow:6px 6px 0 var(--ink);padding:1rem 1.2rem;margin-bottom:.9rem;gap:0;}
[class*="st-key-kind-active"]{background:var(--ink);box-shadow:6px 6px 0 var(--red);}
[class*="st-key-kind-"] [data-testid="stElementContainer"]{position:static;}
[class*="st-key-kind-"] [data-testid="stButton"]{position:absolute;inset:0;}
[class*="st-key-kind-"] [data-testid="stButton"]>button{width:100%;height:100%;opacity:0 !important;box-shadow:none !important;}
.kind-in{display:flex;justify-content:space-between;align-items:baseline;gap:.6rem;}
.kind-l{font-family:'Archivo Black',sans-serif;font-size:1.45rem;color:var(--ink);}
.kind-n{font-family:'Special Elite',monospace;font-size:.72rem;color:var(--ink);white-space:nowrap;}
[class*="st-key-kind-active"] .kind-l{color:var(--lime);}
[class*="st-key-kind-active"] .kind-n{color:var(--cream);}

.st-key-paperlist{position:relative;background:var(--cream);border:2px solid var(--ink);border-radius:2px;
box-shadow:10px 10px 0 var(--ink);padding:1.6rem 1.8rem .6rem;gap:0;}
.st-key-paperlist::before{content:"";position:absolute;top:-15px;left:46px;width:84px;height:26px;
background:rgba(255,213,66,.9);border:1px solid rgba(0,0,0,.12);transform:rotate(-2deg);}
.plist-h{font-family:'Archivo Black',sans-serif;font-size:1.6rem;color:var(--ink);padding-bottom:.9rem;
border-bottom:2px dashed var(--ink);}
[class*="st-key-prow-"]{border-bottom:2px dashed var(--ink);padding:.85rem 0;gap:.4rem;}
.prow-main{display:flex;gap:.9rem;align-items:center;}
.ftype{flex:none;width:40px;height:52px;background:#fff;border:2px solid var(--ink);border-radius:3px;
display:flex;align-items:flex-end;justify-content:center;padding-bottom:3px;}
.ftype span{font-weight:700;font-size:.6rem;color:var(--red);}
.prow-t{font-family:'Special Elite',monospace;font-size:1.02rem;color:var(--ink);line-height:1.25;}
.prow-m{font-size:.72rem;color:var(--ink);margin-top:.15rem;}
.sol{font-weight:700;font-size:.58rem;background:var(--blue);border:1.5px solid var(--ink);padding:.05rem .35rem;
margin-left:.6rem;vertical-align:middle;text-transform:uppercase;letter-spacing:.04em;}

[class*="st-key-lnk-view"] a,[class*="st-key-lnk-dl"] a{border:2px solid var(--ink) !important;border-radius:2px !important;
padding:.45rem 1rem !important;min-height:0 !important;box-shadow:none !important;text-decoration:none !important;
justify-content:center;width:100%;}
[class*="st-key-lnk-view"] a{background:var(--cream) !important;}
[class*="st-key-lnk-dl"] a{background:var(--ink) !important;}
[class*="st-key-lnk-view"] a,[class*="st-key-lnk-view"] a p,[class*="st-key-lnk-view"] a div{color:var(--ink) !important;}
[class*="st-key-lnk-dl"] a,[class*="st-key-lnk-dl"] a p,[class*="st-key-lnk-dl"] a div{color:var(--lime) !important;}
[class*="st-key-lnk-view"] a p,[class*="st-key-lnk-dl"] a p{font-weight:700;font-size:.8rem;margin:0;}
[class*="st-key-prow-"] [data-testid="stButton"]>button{background:var(--pink);box-shadow:none;padding:.45rem .9rem;}
[class*="st-key-prow-"] [data-testid="stButton"]>button p{font-size:.8rem;}

.st-key-note-yellow-R-next .note-b{font-size:1.5rem;}
.st-key-cta-next{margin-top:1.1rem;}
.st-key-cta-next [data-testid="stButton"]>button{background:var(--ink) !important;border:2px solid var(--ink);
box-shadow:5px 5px 0 var(--red) !important;width:100%;padding:.8rem 1rem;border-radius:3px;}
.st-key-cta-next [data-testid="stButton"]>button p{color:var(--lime) !important;font-weight:700;margin:0;}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def _blank(value: object) -> bool:
    return value is None or bool(pd.isna(value))


def _flag(value: object) -> bool:
    return False if _blank(value) else bool(value)


def file_type(filename: object) -> str:
    text = str(filename)
    ext = text.rsplit(".", 1)[-1].lower() if "." in text else ""
    return FILE_TYPES.get(ext, ext.upper() or "FILE")


def question_count(value: object) -> int:
    return 0 if _blank(value) else int(value)


def paper_heading(label: object, term: object, year: object) -> str:
    """'Final . Fall 2025'; term and year are left out when the backend does not know them."""
    term_text = "" if _blank(term) else str(term).strip()
    year_text = "" if _blank(year) else str(int(year))
    when = " ".join(part for part in (term_text, year_text) if part)
    return f"{label} {DOT} {when}" if when else str(label)


def paper_meta(questions: object, total_marks: object) -> str:
    n = question_count(questions)
    if n == 0:
        return "scan not readable here"
    parts = [f"{n} question" + ("" if n == 1 else "s")]
    if not _blank(total_marks):
        parts.append(f"{float(total_marks):g} marks")
    return f" {DOT} ".join(parts)


def row_markup(filename: object, heading: str, meta: str, includes_solution: object) -> str:
    solution = '<span class="sol">includes solution</span>' if _flag(includes_solution) else ""
    return (
        '<div class="prow-main">'
        f'<div class="ftype"><span>{esc(file_type(filename))}</span></div>'
        f'<div><div class="prow-t">{esc(heading)}{solution}</div>'
        f'<div class="prow-m">{esc(meta)}</div></div></div>'
    )


def kind_markup(label: object, papers: object) -> str:
    n = int(papers)
    count = f"{n} paper" + ("" if n == 1 else "s")
    return (
        '<div class="kind-in">'
        f'<span class="kind-l">{esc(label)}</span><span class="kind-n">{count}</span></div>'
    )


def crumb_markup(subject: str, kind_label: str) -> str:
    return (
        f'<div class="crumb"><u>Past papers</u> / {esc(subject)} / {esc(kind_label)}</div>'
    )