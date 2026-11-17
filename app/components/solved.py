"""Markup and CSS for the solved paper screen (Upcoming paper, mockup page 7)."""

from __future__ import annotations

import re

import streamlit as st
from theme import DOT, esc

SOURCE_BADGES = {
    "archive": ("From the archive", ""),
    "ai": ("AI-generated: verify", "pink"),
}

CSS = """
[class*="st-key-sq-"]{background:var(--cream);border:2px solid var(--ink);border-radius:2px;
box-shadow:6px 6px 0 var(--ink);padding:1.1rem 1.4rem 1.2rem;margin-bottom:1.3rem;gap:.6rem;}
.sq-h{display:flex;justify-content:space-between;align-items:flex-start;gap:.8rem;flex-wrap:wrap;}
.sq-t{font-family:'Archivo Black',sans-serif;font-size:1.2rem;color:var(--ink);}
.sq-badges{display:flex;gap:.4rem;flex-wrap:wrap;align-items:center;}
.qchip.pink{background:var(--pink);}
.stApp [class*="st-key-sq-"] [data-testid="stMarkdownContainer"] :where(p,li){font-family:'Special Elite',monospace;
font-size:.95rem;line-height:1.55;color:var(--ink);}
.codebox{background:#111;color:var(--cream);font-family:Consolas,'JetBrains Mono',monospace;font-size:.82rem;
line-height:1.5;padding:1rem 1.1rem;overflow-x:auto;white-space:pre;margin:.4rem 0;}
.sq-note{font-size:.76rem;color:var(--ink);font-style:italic;}
.sq-q{font-family:'Special Elite',monospace;font-size:.9rem;color:var(--ink);white-space:pre-wrap;}
.st-key-sv-up,.st-key-sv-up-top{width:100% !important;}
.st-key-sv-up [data-testid="stElementContainer"],.st-key-sv-up [data-testid="stButton"],
.st-key-sv-up-top [data-testid="stElementContainer"],.st-key-sv-up-top [data-testid="stButton"]{width:100% !important;}
.stApp .st-key-sv-up button,.stApp .st-key-sv-up-top button{width:100%;background:var(--ink) !important;
border:2px solid var(--ink);border-radius:3px;box-shadow:5px 5px 0 var(--red) !important;padding:.8rem 1rem;}
.stApp .st-key-sv-up-top button{box-shadow:none !important;padding:.5rem 1rem;}
.stApp .st-key-sv-up button p,.stApp .st-key-sv-up-top button p{color:var(--lime) !important;font-weight:700;}
.st-key-sv-pdf,.st-key-sv-pdf [data-testid="stElementContainer"],.st-key-sv-pdf [data-testid="stDownloadButton"]{width:100% !important;}
.stApp .st-key-sv-pdf button{background:transparent !important;border:0 !important;box-shadow:none !important;
text-decoration:underline;}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


_FENCE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.S)


def split_blocks(text: str) -> list[tuple[str, str]]:
    """Split markdown into ('text', ...) and ('code', ...) parts at ``` fences."""
    parts: list[tuple[str, str]] = []
    pos = 0
    for match in _FENCE.finditer(text):
        before = text[pos:match.start()]
        if before.strip():
            parts.append(("text", before))
        parts.append(("code", match.group(1).rstrip("\n")))
        pos = match.end()
    if text[pos:].strip():
        parts.append(("text", text[pos:]))
    return parts


def code_markup(code: str) -> str:
    return f'<pre class="codebox"><code>{esc(code)}</code></pre>'


def card_header(label: str, topic: object, marks: object, leetcode: bool, source: str) -> str:
    title = f"{label} {DOT} {topic}" if topic else label
    badges = []
    if leetcode:
        badges.append('<span class="qchip blue">LeetCode-style</span>')
    if marks is not None and marks == marks:
        badges.append(f'<span class="qchip">{float(marks):g} marks</span>')
    text, tone = SOURCE_BADGES.get(source, ("", ""))
    if text:
        badges.append(f'<span class="qchip {tone}">{esc(text)}</span>')
    return (
        f'<div class="sq-h"><span class="sq-t">{esc(title)}</span>'
        f'<span class="sq-badges">{"".join(badges)}</span></div>'
    )