"""CSS and helpers for the upload card (2) and the syllabus card (3) of the Upcoming paper form."""

from __future__ import annotations

import re

import streamlit as st
from theme import esc

FILE_TYPES = ["pdf", "docx", "txt", "jpg", "jpeg", "png"]
KINDS = ["quiz", "mid", "final"]
KIND_LABELS = {"quiz": "Quiz", "mid": "Mid", "final": "Final"}

CSS = """
.card-note{font-family:'Special Elite',monospace;font-size:.82rem;color:var(--ink);}
[data-testid="stFileUploaderDropzone"]{background:#fff;border:2px dashed var(--ink);border-radius:2px;}
[data-testid="stFileUploaderDropzone"] button{background:var(--ink) !important;border:2px solid var(--ink);border-radius:2px;}
[data-testid="stFileUploaderDropzone"] button,[data-testid="stFileUploaderDropzone"] button *{color:var(--lime) !important;font-weight:700;}
.fname{font-family:'Special Elite',monospace;font-size:.85rem;color:var(--ink);word-break:break-all;}
.ychip{display:inline-block;background:var(--yellow);border:2px solid var(--ink);padding:.25rem .6rem;
font-size:.8rem;font-weight:700;margin:0 .5rem .5rem 0;}
.found{font-size:.85rem;color:var(--ink);margin:.2rem 0 .4rem;}

[class*="st-key-topic-"],[class*="st-key-topic-"] [data-testid="stElementContainer"],
[class*="st-key-topic-"] [data-testid="stButton"]{width:100% !important;}
[class*="st-key-topic-"] [data-testid="stButton"]>button{background:var(--pink);box-shadow:none;border-radius:2px;
width:100%;padding:.35rem .6rem;}
[class*="st-key-topic-"] [data-testid="stButton"]>button p{font-size:.78rem;white-space:normal;}
.st-key-up_topic_in [data-baseweb="input"]{background:#fff !important;border:2px solid var(--ink) !important;border-radius:2px !important;}
.st-key-up_topic_in [data-baseweb="base-input"]{background:#fff !important;}
.st-key-up_topic_add [data-testid="stButton"]>button{background:var(--ink) !important;box-shadow:none;border-radius:2px;width:100%;}
.st-key-up_topic_add [data-testid="stButton"]>button p{color:var(--lime) !important;}
.st-key-up_topic_add,.st-key-up_topic_add [data-testid="stElementContainer"],
.st-key-up_topic_add [data-testid="stButton"]{width:100% !important;}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def guess_year(name: str, current: int) -> int:
    """Year found in a file name such as Mid1_Fall2025.pdf, else the current year."""
    match = re.search(r"(20\d{2})", name)
    year = int(match.group(1)) if match else current
    return year if 2000 <= year <= current + 1 else current


def guess_kind(name: str) -> str:
    low = name.lower()
    if "quiz" in low:
        return "quiz"
    if "final" in low:
        return "final"
    return "mid"


def count_of(value: object) -> int:
    """Number of items, whether the backend returns a count or a list."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def chips_markup(names: list[str]) -> str:
    return "".join(f'<span class="ychip">{esc(name)} \u2713</span>' for name in names)