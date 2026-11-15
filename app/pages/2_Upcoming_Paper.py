import bootstrap  # noqa: F401
from datetime import date

import state
import streamlit as st
import theme
from common import get_generator, get_library, get_predictor
from components import predict_form as form
from src.inference.errors import LowConfidenceError
from src.inference.library import default_duration

theme.setup("Upcoming paper", active="upcoming")
form.inject_css()
lib = get_library()
pred = get_predictor()

YEAR = date.today().year
ALL_SUBJECTS = lib.all_subjects().sort_values(["papers", "subject"], ascending=[False, True])
SUPPORTED = set(pred.subjects())
ALL_NAMES = list(ALL_SUBJECTS["subject"])
TOP = [name for name in ALL_NAMES if name in SUPPORTED][:3]

st.session_state.setdefault("up_screen", "form")
st.session_state.setdefault("up_paper", None)
st.session_state.setdefault("up_subject", TOP[0] if TOP else None)
st.session_state.setdefault("up_exam", "final")
st.session_state.setdefault("up_minutes", default_duration("final"))
st.session_state.setdefault("up_custom", False)
st.session_state.setdefault("up_timer_on", True)

# Subject and exam chosen on the Past papers page ("Predict the next one").
_pre_subject, _pre_exam = state.take_preselection()
if _pre_subject:
    st.session_state["up_subject"] = _pre_subject
    st.session_state["up_more_sel"] = None
    st.session_state["up_screen"] = "form"
if _pre_exam:
    st.session_state["up_exam"] = _pre_exam
    st.session_state["up_minutes"] = default_duration(_pre_exam)
    st.session_state["up_custom"] = False


def pick_subject(name: str) -> None:
    st.session_state["up_subject"] = name
    st.session_state["up_more_sel"] = None


def pick_more() -> None:
    chosen = st.session_state.get("up_more_sel")
    if chosen:
        st.session_state["up_subject"] = chosen


def choose_exam(exam: str) -> None:
    st.session_state["up_exam"] = exam
    st.session_state["up_minutes"] = default_duration(exam)
    st.session_state["up_custom"] = False


def choose_minutes(minutes: int) -> None:
    st.session_state["up_minutes"] = minutes
    st.session_state["up_custom"] = False


def choose_custom() -> None:
    st.session_state["up_custom"] = True


def toggle_timer() -> None:
    st.session_state["up_timer_on"] = not st.session_state["up_timer_on"]


def back_to_form() -> None:
    st.session_state["up_screen"] = "form"


def effective_minutes() -> int:
    if st.session_state["up_custom"]:
        return int(st.session_state.get("up_custom_min", 120))
    return int(st.session_state["up_minutes"])


def unsupported_note(subject: str) -> None:
    reason = None
    try:
        reason = lib.overview(subject).get("reason_if_unsupported")
    except Exception:
        pass
    theme.notice(reason or "Prediction is not available for this subject yet. You can still read its past papers.")
    st.page_link("pages/1_Past_Papers.py", label="Browse past papers " + theme.ARROW)


def subject_exam_card() -> None:
    subject = st.session_state["up_subject"]
    with st.container(key="card-1"):
        st.markdown(form.card_heading(1, "Subject and exam"), unsafe_allow_html=True)
        names = TOP + ([subject] if subject and subject not in TOP else [])
        cols = st.columns(form.chip_widths(names), gap="small")
        for i, (col, name) in enumerate(zip(cols, names)):
            mode = "on" if name == subject else "off"
            with col:
                with st.container(key=f"chip-{mode}-subj{i}"):
                    st.button(name, key=f"up_subj_{i}", on_click=pick_subject, args=(name,))
        st.selectbox(
            "More subjects",
            ALL_NAMES,
            index=None,
            placeholder="More subjects: search all subjects",
            key="up_more_sel",
            on_change=pick_more,
            label_visibility="collapsed",
            format_func=lambda s: s if s in SUPPORTED else f"{s} (past papers only)",
        )
        if subject and subject not in SUPPORTED:
            unsupported_note(subject)

        exam = st.session_state["up_exam"]
        cols = st.columns(4, gap="small")
        for i, (col, (value, label)) in enumerate(zip(cols, form.EXAMS)):
            mode = "on" if value == exam else "off"
            with col:
                with st.container(key=f"exam-{mode}-{i}"):
                    st.button(label, key=f"up_exam_{value}", on_click=choose_exam, args=(value,))
        st.radio("Year of the exam", [YEAR, YEAR + 1], horizontal=True, key="up_year")


def timer_card() -> None:
    timer_on = st.session_state["up_timer_on"]
    with st.container(key="card-4"):
        st.markdown(form.card_heading(4, "Timer", optional=True), unsafe_allow_html=True)
        labels = ["Timer ON" if timer_on else "Timer OFF"] + [label for _, label in form.DURATIONS] + ["Custom"]
        cols = st.columns(form.chip_widths(labels), gap="small")
        with cols[0]:
            with st.container(key=f"chip-{'on' if timer_on else 'off'}-timer"):
                st.button(labels[0], key="up_timer_toggle", on_click=toggle_timer)
        for col, (minutes, label) in zip(cols[1:5], form.DURATIONS):
            picked = (not st.session_state["up_custom"]) and st.session_state["up_minutes"] == minutes
            with col:
                with st.container(key=f"pick-{'on' if picked else 'off'}-{minutes}"):
                    st.button(label, key=f"up_dur_{minutes}", on_click=choose_minutes, args=(minutes,), disabled=not timer_on)
        with cols[5]:
            with st.container(key=f"pick-{'on' if st.session_state['up_custom'] else 'off'}-custom"):
                st.button("Custom", key="up_dur_custom", on_click=choose_custom, disabled=not timer_on)
        if timer_on and st.session_state["up_custom"]:
            st.number_input("Minutes", min_value=5, max_value=360, value=120, step=5, key="up_custom_min")


def difficulty_note(subject: str | None, exam: str) -> None:
    pct = None
    if subject in SUPPORTED:
        try:
            pct = form.difficulty_percentages(lib.blueprint(subject, exam).get("difficulty_mix"))
        except Exception:
            pct = None
    with st.container(key="note-yellow-R-diff"):
        body = "Matched to past papers, including LeetCode-style coding questions."
        st.markdown(
            f'<div class="note-t">Difficulty, the FAST way</div><div class="note-b">{body}</div>',
            unsafe_allow_html=True,
        )
        if pct:
            st.markdown(form.difficulty_markup(pct), unsafe_allow_html=True)
            st.markdown('<div class="dnote">Difficulty is a proxy built from past papers.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="dnote">Pick a supported subject and exam to see the mix.</div>', unsafe_allow_html=True)


def generate(subject: str, exam: str, advanced: dict) -> str | None:
    """Build the paper; returns an error message, or None when the paper is stored."""
    try:
        with st.spinner("Reading the archive and building your paper..."):
            paper = get_generator().generate(
                subject,
                exam,
                int(st.session_state["up_year"]),
                model=advanced["model"],
                syllabus_items=None,
                extra=None,
                n_questions=advanced["n_questions"] or None,
                total_marks=advanced["total_marks"] or None,
                explore_share=advanced["explore_share"],
            )
    except LowConfidenceError as err:
        return str(err)
    except Exception:
        return "Something went wrong while building the paper. Please try again."
    if st.session_state["up_timer_on"]:
        paper["duration_minutes"] = effective_minutes()
    st.session_state["up_paper"] = paper
    st.session_state["up_screen"] = "paper"
    st.rerun()
    return None


def right_column() -> None:
    subject = st.session_state["up_subject"]
    exam = st.session_state["up_exam"]
    difficulty_note(subject, exam)
    theme.note(
        "also", "",
        "Also included: a few less-predicted syllabus areas, so surprise questions do not catch you out.",
        "pink", "L",
    )
    with st.container(key="cta-gen"):
        clicked = st.button(
            "Generate predicted paper " + theme.ARROW,
            key="up_generate",
            disabled=subject not in SUPPORTED,
        )
    st.markdown(
        '<div class="fine">Predicted questions are real past questions chosen for the topics most likely to come.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Advanced"):
        advanced = {
            "model": st.selectbox("Model", pred.model_names(), key="up_model"),
            "n_questions": int(st.number_input("Number of questions (0 = automatic)", 0, 30, 0, key="up_nq")),
            "total_marks": int(st.number_input("Total marks (0 = automatic)", 0, 500, 0, step=5, key="up_marks")),
            "explore_share": float(st.slider("Share of less-predicted questions", 0.0, 0.5, 0.2, 0.05, key="up_explore")),
        }
    if clicked:
        error = generate(subject, exam, advanced)
        if error:
            theme.notice(error)


def form_screen() -> None:
    theme.title("Prediction desk", "Predict the", "next paper")
    left, right = st.columns([2.4, 1], gap="large")
    with left:
        subject_exam_card()
        timer_card()
    with right:
        right_column()


def paper_screen(paper: dict) -> None:
    st.button("\u2190 Change inputs", key="up_back_form", on_click=back_to_form)
    theme.title_last("Tentative paper", paper["title"])
    st.write(f"{len(paper['questions'])} questions generated. The predicted paper screen is built in the next step.")
    for line in paper.get("notes", []):
        st.caption(line)


if st.session_state["up_screen"] == "paper" and st.session_state["up_paper"]:
    paper_screen(st.session_state["up_paper"])
else:
    form_screen()