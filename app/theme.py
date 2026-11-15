"""Look and feel of SpillThePaper: CSS, top navigation and small HTML helpers."""

from __future__ import annotations

import html

import streamlit as st

ARROW = "\u2192"
DOT = "\u00b7"

PAGES = [
    ("home", "Home", "Home.py"),
    ("past", "Past papers", "pages/1_Past_Papers.py"),
    ("upcoming", "Upcoming paper", "pages/2_Upcoming_Paper.py"),
    ("results", "My results", "pages/3_My_Results.py"),
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Caveat:wght@700&family=Space+Grotesk:wght@400;500;700&family=Special+Elite&display=swap');
:root{--sage:#BED3B4;--cream:#FCF6E4;--ink:#111111;--lime:#D9FF3A;--red:#D4281A;
--yellow:#FFD542;--pink:#FF9CC7;--blue:#8FD6FF;--orange:#FFB347;}
.stApp{background:var(--sage);color:var(--ink);}
.stApp :where(p,label,button,input,textarea,li,a,div,h1,h2,h3,h4){font-family:'Space Grotesk',system-ui,sans-serif;}
.stApp .stMarkdown p,.stApp button p,.stApp [data-testid="stPageLink"] a p{font-family:'Space Grotesk',system-ui,sans-serif;}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],#MainMenu,footer{display:none !important;}
[data-testid="stElementContainer"]:has(style){display:none;}
[data-testid="stMainBlockContainer"]{max-width:1240px;padding:1.4rem 2rem 4rem;}
[data-testid="stMarkdown"],[data-testid="stMarkdownContainer"]{margin-bottom:0 !important;}
.st-key-topnav [data-testid="stHorizontalBlock"]{gap:.5rem !important;}

.logo{font-family:'Archivo Black',sans-serif;font-size:1.45rem;letter-spacing:.02em;color:var(--ink);}
.st-key-topnav [data-testid="stPageLink"] a{background:var(--cream);border:2px solid var(--ink);
border-radius:10px;padding:.35rem .9rem;justify-content:center;min-height:0;text-decoration:none;}
.st-key-topnav [data-testid="stPageLink"] a p{font-weight:700;font-size:.82rem;color:var(--ink);
white-space:nowrap;margin:0;}

.kicker{font-family:'Special Elite',monospace;font-size:.72rem;letter-spacing:.18em;
text-transform:uppercase;color:var(--ink);margin-bottom:.15rem;}
.sp-title{font-family:'Archivo Black',sans-serif;font-size:clamp(2.4rem,5vw,3.6rem);
line-height:1.05;color:var(--ink);margin:0 0 .6rem;}
.hl{background:var(--yellow);padding:0 .22em;box-decoration-break:clone;-webkit-box-decoration-break:clone;}

.st-key-hero{position:relative;background:var(--cream);border:2px solid var(--ink);border-radius:3px;
box-shadow:8px 8px 0 var(--ink);transform:rotate(-1deg);padding:2rem 2.4rem 2.2rem;margin-top:.9rem;}
.st-key-hero::before{content:"";position:absolute;top:-16px;left:34px;width:96px;height:28px;
background:rgba(255,213,66,.9);border:1px solid rgba(0,0,0,.12);transform:rotate(-3deg);}
.hero-top{display:flex;justify-content:space-between;align-items:center;gap:1rem;}
.stamp{border:3px solid var(--red);color:var(--red);font-family:'Archivo Black',sans-serif;
font-size:1.5rem;letter-spacing:.12em;padding:.15rem .8rem;border-radius:4px;
transform:rotate(-3deg);margin-top:-.3rem;white-space:nowrap;}
.hero-h{font-family:'Archivo Black',sans-serif;font-size:clamp(2.6rem,5.4vw,4.4rem);line-height:1.02;
color:var(--ink);margin:.5rem 0 1rem;}
.hero-p{font-family:'Special Elite',monospace;font-size:1.02rem;line-height:1.55;max-width:31rem;color:var(--ink);}
.redact{height:15px;background:var(--ink);margin:.55rem 0;}

.st-key-cta-predict [data-testid="stPageLink"] a{background:var(--ink) !important;border:2px solid var(--ink);
border-radius:3px;box-shadow:5px 5px 0 var(--red);padding:.7rem 1.1rem;justify-content:center;text-decoration:none;}
.st-key-cta-predict [data-testid="stPageLink"] a p{color:var(--lime) !important;font-weight:700;margin:0;}
.st-key-cta-browse [data-testid="stPageLink"] a{background:var(--cream) !important;border:2px solid var(--ink);
border-radius:3px;padding:.7rem 1.1rem;justify-content:center;text-decoration:none;}
.st-key-cta-browse [data-testid="stPageLink"] a p{color:var(--ink) !important;font-weight:700;margin:0;}

[class*="st-key-note-"]{border:2px solid var(--ink);border-radius:2px;box-shadow:5px 5px 0 var(--ink);
padding:.9rem 1.2rem 1rem;position:relative;gap:.1rem;}
[class*="st-key-note-yellow"]{background:var(--yellow);}
[class*="st-key-note-pink"]{background:var(--pink);}
[class*="st-key-note-blue"]{background:var(--blue);}
[class*="st-key-note-orange"]{background:var(--orange);}
[class*="st-key-note-"][class*="-R-"]{transform:rotate(1deg);}
[class*="st-key-note-"][class*="-L-"]{transform:rotate(-1.2deg);}
[class*="st-key-note-"] [data-testid="stElementContainer"]{position:static;}
[class*="st-key-note-"] [data-testid="stPageLink"]{position:absolute;inset:0;}
[class*="st-key-note-"] [data-testid="stPageLink"] a{width:100%;height:100%;opacity:0;}
.note-t{font-family:'Archivo Black',sans-serif;font-size:1rem;color:var(--ink);}
.note-b{font-family:'Caveat',cursive;font-weight:700;font-size:1.25rem;line-height:1.15;color:var(--ink);}

.steps{display:grid;grid-template-columns:repeat(5,1fr);gap:1.6rem;background:var(--ink);border-radius:6px;
padding:1.3rem 1.8rem 1.5rem;box-shadow:6px 6px 0 var(--red);margin-top:2.4rem;}
.step-n{font-family:'Archivo Black',sans-serif;font-size:2.1rem;color:var(--lime);line-height:1;margin-bottom:.35rem;}
.step-t{color:var(--cream);font-size:.86rem;line-height:1.35;}
@media (max-width:900px){.steps{grid-template-columns:repeat(2,1fr);}}
"""


def esc(text: object) -> str:
    """Escape text coming from the backend before it goes into HTML."""
    return html.escape(str(text))


def _active_css(active: str) -> str:
    return (
        f'.st-key-topnav .st-key-nav_{active} [data-testid="stPageLink"] a'
        "{background:#111111 !important;}"
        f'.st-key-topnav .st-key-nav_{active} [data-testid="stPageLink"] a p'
        "{color:#D9FF3A !important;}"
    )


def top_nav() -> None:
    with st.container(key="topnav"):
        cols = st.columns([5.4, 0.85, 1.25, 1.55, 1.2], vertical_alignment="center")
        cols[0].markdown('<div class="logo">SPILL THE PAPER</div>', unsafe_allow_html=True)
        for col, (key, label, path) in zip(cols[1:], PAGES):
            with col:
                with st.container(key=f"nav_{key}"):
                    st.page_link(path, label=label)


def setup(page_title: str, active: str) -> None:
    """First call on every page: page config, CSS and the top navigation."""
    st.set_page_config(
        page_title=f"{page_title} | SpillThePaper",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(f"<style>{CSS}{_active_css(active)}</style>", unsafe_allow_html=True)
    top_nav()


def title(kicker: str, plain: str, highlighted: str) -> None:
    """Small typewriter label and a big title with a highlighted last word."""
    st.markdown(
        f'<div class="kicker">{esc(kicker)}</div>'
        f'<div class="sp-title">{esc(plain)} <span class="hl">{esc(highlighted)}</span></div>',
        unsafe_allow_html=True,
    )


def note(key: str, heading: str, body: str, color: str, tilt: str, target: str | None = None) -> None:
    """A tilted sticky note (color: yellow/pink/blue/orange, tilt: L or R).
    With a target page the whole note is clickable."""
    with st.container(key=f"note-{color}-{tilt}-{key}"):
        st.markdown(
            f'<div class="note-t">{esc(heading)}</div><div class="note-b">{esc(body)}</div>',
            unsafe_allow_html=True,
        )
        if target:
            st.page_link(target, label=heading)


def steps_strip(items: list[str]) -> None:
    cells = "".join(
        f'<div><div class="step-n">{i:02d}</div><div class="step-t">{esc(text)}</div></div>'
        for i, text in enumerate(items, 1)
    )
    st.markdown(f'<div class="steps">{cells}</div>', unsafe_allow_html=True)