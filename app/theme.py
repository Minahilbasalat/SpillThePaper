"""Look and feel of SpillThePaper: CSS, top navigation and small HTML helpers."""

from __future__ import annotations

import html

import streamlit as st

ARROW = "\u2192"
DOT = "\u00b7"
FOLDER_COLORS = ["yellow", "pink", "blue", "orange"]

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
border-radius:10px;padding:.35rem .9rem;justify-content:center;min-height:0;text-decoration:none;width:100%;}
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

.stApp [data-testid="stButton"]>button{background:var(--cream);border:2px solid var(--ink);border-radius:3px;
color:var(--ink);box-shadow:4px 4px 0 var(--ink);padding:.5rem 1.1rem;min-height:0;}
.stApp [data-testid="stButton"]>button p{font-weight:700;color:var(--ink);margin:0;}
.stApp [data-testid="stButton"]>button:hover{background:var(--yellow);border-color:var(--ink);color:var(--ink);}

[data-testid="InputInstructions"]{display:none;}
.st-key-past_search [data-baseweb="input"]{background:var(--cream) !important;border:2px solid var(--ink) !important;
border-radius:2px !important;box-shadow:5px 5px 0 var(--ink);}
.st-key-past_search [data-baseweb="base-input"]{background:transparent !important;}
.st-key-past_search input{background:transparent url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23111111' stroke-width='3'%3E%3Ccircle cx='10' cy='10' r='7'/%3E%3Cpath d='M15 15l6 6'/%3E%3C/svg%3E") no-repeat 14px center !important;
padding:.75rem .8rem .75rem 2.8rem !important;font-size:.95rem;color:var(--ink);}

[class*="st-key-folder-"]{position:relative;background:var(--cream);border:2px solid var(--ink);border-radius:3px;
box-shadow:6px 6px 0 var(--ink);padding:1rem 1.2rem .9rem;margin-top:1.4rem;gap:0;}
[class*="st-key-folder-"]::before{content:"";position:absolute;top:-22px;left:-2px;width:104px;height:22px;
border:2px solid var(--ink);border-bottom:0;border-radius:8px 8px 0 0;background:var(--yellow);}
[class*="st-key-folder-yellow"]::before{background:var(--yellow);}
[class*="st-key-folder-pink"]::before{background:var(--pink);}
[class*="st-key-folder-blue"]::before{background:var(--blue);}
[class*="st-key-folder-orange"]::before{background:var(--orange);}
[class*="st-key-folder-"]:hover{transform:translate(-2px,-2px);box-shadow:8px 8px 0 var(--ink);}
[class*="st-key-folder-"] [data-testid="stElementContainer"]{position:static;}
[class*="st-key-folder-"] [data-testid="stButton"]{position:absolute;inset:0;}
[class*="st-key-folder-"] [data-testid="stButton"]>button{width:100%;height:100%;opacity:0 !important;box-shadow:none !important;}
.folder-body{min-height:9.2rem;display:flex;flex-direction:column;justify-content:space-between;}
.folder-name{font-family:'Archivo Black',sans-serif;font-size:1.3rem;line-height:1.15;color:var(--ink);}
.folder-kinds{font-family:'Special Elite',monospace;font-size:.72rem;color:var(--ink);}
.folder-count{font-family:'Special Elite',monospace;font-size:.72rem;background:var(--yellow);padding:0 .15rem;}

.st-key-banner{background:var(--pink);border:2px solid var(--ink);border-radius:2px;box-shadow:6px 6px 0 var(--ink);
transform:rotate(-.7deg);padding:1.2rem 1.6rem;margin-top:2.4rem;}
.banner-t{font-family:'Caveat',cursive;font-weight:700;font-size:1.6rem;color:var(--ink);}
.st-key-cta-banner [data-testid="stPageLink"] a{background:var(--ink) !important;border:2px solid var(--ink);
border-radius:3px;padding:.65rem 1.1rem;justify-content:center;text-decoration:none;}
.st-key-cta-banner [data-testid="stPageLink"] a p{color:var(--lime) !important;font-weight:700;margin:0;}

.notice{background:var(--cream);border:2px dashed var(--ink);padding:1rem 1.3rem;
font-family:'Special Elite',monospace;font-size:.95rem;color:var(--ink);margin:1rem 0;}
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


def title_last(kicker: str, text: str) -> None:
    """Like title(), highlighting the last word of a backend-provided text."""
    head, _, last = text.rpartition(" ")
    title(kicker, head, last)


def notice(text: str) -> None:
    st.markdown(f'<div class="notice">{esc(text)}</div>', unsafe_allow_html=True)


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


def folder_markup(name: str, kinds: str, papers: int) -> str:
    """Inside of a subject folder card: name on top, exam kinds and paper count below."""
    n = int(papers)
    label = f"{n} paper" + ("" if n == 1 else "s")
    return (
        '<div class="folder-body">'
        f'<div class="folder-name">{esc(name)}</div>'
        f'<div><div class="folder-kinds">{esc(kinds)}</div>'
        f'<span class="folder-count">{label}</span></div></div>'
    )