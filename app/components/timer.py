"""Exam timer: state helpers, CSS and the live clock (updates every second)."""

from __future__ import annotations

import math
import time

import streamlit as st

END = "ex_end"        # epoch seconds when the exam ends (while running)
LEFT = "ex_left"      # seconds left (while paused), else None
TOTAL = "ex_total"    # total seconds of this exam
HIDDEN = "ex_hidden"  # clock hidden by the student

AMBER_AT = 600  # 10 minutes
RED_AT = 300    # 5 minutes

CSS = """
.exam-badge{display:inline-block;background:var(--yellow);border:2px solid var(--ink);padding:.3rem .7rem;
font-family:'Special Elite',monospace;font-size:.85rem;color:var(--ink);}
.st-key-exam-card{position:relative;background:var(--cream);border:2px solid var(--ink);border-radius:2px;
box-shadow:8px 8px 0 var(--ink);padding:1.8rem 2rem 1.4rem;margin-top:.9rem;gap:.2rem;}
.st-key-exam-card::before{content:"";position:absolute;top:-16px;left:34px;width:96px;height:28px;
background:rgba(255,213,66,.9);border:1px solid rgba(0,0,0,.12);transform:rotate(-3deg);}

.st-key-clock-panel{background:var(--ink);border:2px solid var(--ink);border-radius:2px;box-shadow:8px 8px 0 var(--red);
padding:1.2rem 1.4rem 1.4rem;gap:.6rem;margin-top:.9rem;}
.clock-l{font-family:'Special Elite',monospace;font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;color:var(--cream);}
.clock-t{font-family:'Archivo Black',sans-serif;font-size:clamp(2.6rem,5vw,4.2rem);line-height:1;color:var(--lime);}
.clock-t.amber{color:var(--orange);}
.clock-t.red{color:#FF5A4A;}
.clock-t.off{color:#777;}
.cbar{height:12px;background:#333;}
.cbar span{display:block;height:100%;background:var(--lime);}
.cbar.amber span{background:var(--orange);}
.cbar.red span{background:#FF5A4A;}
.clock-s{font-size:.78rem;color:var(--cream);}
.clock-up{background:var(--red);color:#fff;font-family:'Archivo Black',sans-serif;padding:.6rem .8rem;
border:2px solid var(--cream);font-size:1rem;}
.clock-pause{font-family:'Special Elite',monospace;font-size:.75rem;color:var(--cream);}
.st-key-clock-panel [data-testid="stElementContainer"],.st-key-clock-panel [data-testid="stButton"]{width:100% !important;}
.stApp .st-key-clock-panel [data-testid="stButton"]>button{width:100%;box-shadow:none;border-radius:2px;padding:.7rem .5rem;}
.stApp .st-key-clock-pause button{background:var(--cream) !important;}
.stApp .st-key-clock-pause button p{color:var(--ink) !important;}
.stApp .st-key-clock-hide button{background:var(--ink) !important;border:2px solid var(--cream) !important;}
.stApp .st-key-clock-hide button p{color:var(--cream) !important;}
.st-key-timer-off{background:var(--cream);border:2px dashed var(--ink);padding:1rem 1.2rem;margin-top:.9rem;
font-family:'Special Elite',monospace;font-size:.9rem;color:var(--ink);}

.st-key-ex-finish-up,.st-key-ex-finish-up [data-testid="stElementContainer"],.st-key-ex-finish-up [data-testid="stButton"],
.st-key-ex-finish,.st-key-ex-finish [data-testid="stElementContainer"],.st-key-ex-finish [data-testid="stButton"]{width:100% !important;}
.stApp .st-key-ex-finish-up button{width:100%;background:var(--red) !important;border:2px solid var(--ink);border-radius:3px;
box-shadow:4px 4px 0 var(--ink) !important;padding:.9rem 1rem;}
.stApp .st-key-ex-finish-up button p{color:#fff !important;font-weight:700;}
.stApp .st-key-ex-finish button{width:100%;border-radius:3px;padding:.7rem 1rem;}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def hhmmss(seconds: int) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def start(total_seconds: int) -> None:
    st.session_state[TOTAL] = int(total_seconds)
    st.session_state[END] = time.time() + int(total_seconds)
    st.session_state[LEFT] = None
    st.session_state[HIDDEN] = False


def reset() -> None:
    for key in (END, LEFT, TOTAL, HIDDEN):
        st.session_state.pop(key, None)


def active() -> bool:
    return int(st.session_state.get(TOTAL, 0) or 0) > 0


def paused() -> bool:
    return st.session_state.get(LEFT) is not None


def remaining() -> int:
    left = st.session_state.get(LEFT)
    if left is not None:
        return int(left)
    end = st.session_state.get(END)
    if end is None:
        return 0
    return max(0, int(math.ceil(end - time.time())))


def toggle_pause() -> None:
    if paused():
        st.session_state[END] = time.time() + int(st.session_state[LEFT])
        st.session_state[LEFT] = None
    else:
        st.session_state[LEFT] = remaining()
        st.session_state[END] = None


def toggle_hidden() -> None:
    st.session_state[HIDDEN] = not st.session_state.get(HIDDEN, False)


def tone(seconds: int) -> str:
    if seconds <= RED_AT:
        return "red"
    if seconds <= AMBER_AT:
        return "amber"
    return ""


@st.fragment(run_every="1s")
def clock() -> None:
    """The black TIME LEFT panel. Only this fragment reruns every second."""
    total = int(st.session_state.get(TOTAL, 0) or 0)
    if total <= 0:
        return
    left = remaining()
    hidden = bool(st.session_state.get(HIDDEN, False))
    is_paused = paused()
    done = left <= 0
    mood = tone(left)
    used = min(100, max(0, round(100 * (total - left) / total)))
    shown = "--:--:--" if hidden else hhmmss(left)
    time_class = "off" if hidden else mood

    with st.container(key="clock-panel"):
        st.markdown(
            '<div class="clock-l">Time left</div>'
            f'<div class="clock-t {time_class}">{shown}</div>'
            f'<div class="cbar {mood}"><span style="width:{used}%"></span></div>'
            f'<div class="clock-s">of {hhmmss(total)} \u00b7 {used}% used</div>',
            unsafe_allow_html=True,
        )
        if done:
            st.markdown('<div class="clock-up">Time is up. Put your pen down.</div>', unsafe_allow_html=True)
        elif is_paused:
            st.markdown('<div class="clock-pause">Paused. The clock is stopped.</div>', unsafe_allow_html=True)
        c_pause, c_hide = st.columns(2, gap="small")
        with c_pause:
            with st.container(key="clock-pause"):
                st.button("Resume" if is_paused else "Pause", key="ex_pause", on_click=toggle_pause, disabled=done)
        with c_hide:
            with st.container(key="clock-hide"):
                st.button("Show clock" if hidden else "Hide clock", key="ex_hide", on_click=toggle_hidden)