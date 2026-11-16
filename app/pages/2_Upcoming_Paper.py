import bootstrap  # noqa: F401
import re
from datetime import date

import state
import streamlit as st
import theme
from common import get_generator, get_library, get_predictor, get_solutions
from results_state import open_results
from components import inputs
from components import predicted
from components import solved
from components import timer
from components import predict_form as form
from src.inference.documents import extract_text
from src.inference.export import paper_to_pdf
from src.inference.errors import LowConfidenceError
from src.inference.library import default_duration
from src.inference.syllabus import default_scope, parse_syllabus
from src.inference.uploads import ingest_papers

theme.setup("Upcoming paper", active="upcoming")
form.inject_css()
inputs.inject_css()
predicted.inject_css()
timer.inject_css()
solved.inject_css()
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
st.session_state.setdefault("up_topics", [])
st.session_state.setdefault("up_upload", None)
st.session_state.setdefault("up_syll_items", [])
st.session_state.setdefault("up_syll_msg", "")
st.session_state.setdefault("up_revealed", [])
st.session_state.setdefault("up_pdf", None)
st.session_state.setdefault("up_exam_timer", True)
st.session_state.setdefault("up_solutions", None)

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


# ---------- callbacks ----------
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


def go(screen: str) -> None:
    st.session_state["up_screen"] = screen


def reveal(index: int) -> None:
    shown = set(st.session_state["up_revealed"])
    shown.add(index)
    st.session_state["up_revealed"] = sorted(shown)


def _add_topics(items: list[str]) -> None:
    topics = st.session_state["up_topics"]
    known = {t.lower() for t in topics}
    for item in items:
        text = item.strip()
        if text and text.lower() not in known:
            topics.append(text)
            known.add(text.lower())


def add_topic() -> None:
    _add_topics([st.session_state.get("up_topic_in", "")])
    st.session_state["up_topic_in"] = ""


def remove_topic(index: int) -> None:
    topics = st.session_state["up_topics"]
    if 0 <= index < len(topics):
        del topics[index]


def add_more_topics() -> None:
    _add_topics(list(st.session_state.get("up_syll_more", [])))
    st.session_state["up_syll_more"] = []


def clear_upload() -> None:
    st.session_state["up_upload"] = None


def effective_minutes() -> int:
    if st.session_state["up_custom"]:
        return int(st.session_state.get("up_custom_min", 120))
    return int(st.session_state["up_minutes"])


# ---------- card 1: subject and exam ----------
def unsupported_note(subject: str) -> None:
    reason = None
    try:
        reason = lib.overview(subject).get("reason_if_unsupported")
    except Exception:
        pass
    detail = f" ({reason})" if reason else ""
    theme.notice(f"Prediction is not available for this subject yet{detail}. You can still read its past papers.")
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


# ---------- card 2: upload recent papers ----------
def ingest(subject: str, files: list, entries: list[dict]) -> None:
    payload = [
        {
            "name": f.name,
            "data": f.getvalue(),
            "year": e["year"],
            "assessment_type": e["type"],
            "number": e["number"],
        }
        for f, e in zip(files, entries)
    ]
    try:
        with st.spinner("Reading your papers (scans can take a moment)..."):
            result = ingest_papers(subject, payload)
    except (ValueError, RuntimeError) as err:
        theme.notice(str(err))
        return
    except Exception:
        theme.notice("These papers could not be read. Try another file or a clearer scan.")
        return
    st.session_state["up_upload"] = {"subject": subject, "result": result, "names": [f.name for f in files]}


def upload_result(subject: str) -> None:
    stored = st.session_state["up_upload"]
    if not stored or stored["subject"] != subject:
        return
    result = stored["result"]
    names = stored["names"]
    accepted = inputs.count_of(getattr(result, "n_papers", 0))
    questions = inputs.count_of(getattr(result, "questions", 0))
    if accepted == len(names):
        st.markdown(inputs.chips_markup(names), unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="found">{accepted} of {len(names)} files accepted. Read the messages below.</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="found">{questions} question(s) found. They are used for this prediction only.</div>',
        unsafe_allow_html=True,
    )
    for message in list(getattr(result, "warnings", []) or []):
        st.caption(str(message))
    st.button("Remove uploaded papers", key="up_clear_upload", on_click=clear_upload)


def upload_card(subject: str | None) -> None:
    supported = subject in SUPPORTED
    with st.container(key="card-2"):
        st.markdown(form.card_heading(2, "Upload recent papers", optional=True), unsafe_allow_html=True)
        st.markdown(
            '<div class="card-note">More recent papers means a sharper prediction.</div>',
            unsafe_allow_html=True,
        )
        files = st.file_uploader(
            "Drop PDFs, photos or docs here",
            type=inputs.FILE_TYPES,
            accept_multiple_files=True,
            key="up_files",
        )
        entries: list[dict] = []
        for i, f in enumerate(files or []):
            c_name, c_year, c_kind, c_num = st.columns([2.6, 1, 1.2, 1.1], vertical_alignment="bottom")
            c_name.markdown(f'<div class="fname">{theme.esc(f.name)}</div>', unsafe_allow_html=True)
            year = c_year.number_input(
                "Year", 2000, YEAR + 1, inputs.guess_year(f.name, YEAR), key=f"up_fy_{i}_{f.name}"
            )
            kind = c_kind.selectbox(
                "Type",
                inputs.KINDS,
                index=inputs.KINDS.index(inputs.guess_kind(f.name)),
                format_func=lambda k: inputs.KIND_LABELS[k],
                key=f"up_fk_{i}_{f.name}",
            )
            number = c_num.number_input("Number (0 = none)", 0, 10, 0, key=f"up_fn_{i}_{f.name}")
            entries.append({"year": int(year), "type": kind, "number": int(number) or None})
        if not supported:
            st.caption("Uploads need a subject that supports prediction.")
        elif files and st.button("Read these papers", key="up_ingest"):
            ingest(subject, list(files), entries)
        if supported:
            upload_result(subject)


# ---------- card 3: syllabus ----------
def read_syllabus(file, exam: str) -> None:
    """Turn an uploaded syllabus into chips (once per file)."""
    sig = f"{file.name}:{file.size}"
    if st.session_state.get("up_syll_sig") == sig:
        return
    st.session_state["up_syll_sig"] = sig
    st.session_state["up_syll_msg"] = ""
    try:
        with st.spinner("Reading the syllabus..."):
            result = extract_text(file.name, file.getvalue())
            parsed = parse_syllabus(result["text"])
            keep = default_scope(parsed, exam)
    except (ValueError, RuntimeError) as err:
        st.session_state["up_syll_msg"] = str(err)
        return
    except Exception:
        st.session_state["up_syll_msg"] = "This syllabus could not be read. Add the topics by hand."
        return
    items = parsed["items"]
    if not items:
        st.session_state["up_syll_msg"] = "No syllabus items were found in this file. Add the topics by hand."
        return
    chosen = [items[i] for i in keep if 0 <= i < len(items)] or items
    _add_topics(chosen)
    st.session_state["up_syll_items"] = items
    message = f"{len(chosen)} of {len(items)} items selected for this exam. Remove what is not included."
    if result.get("warning"):
        message += f" {result['warning']}"
    st.session_state["up_syll_msg"] = message


def topic_chips() -> None:
    topics = st.session_state["up_topics"]
    if not topics:
        st.markdown(
            '<div class="card-note">No syllabus added: every topic of the subject is considered.</div>',
            unsafe_allow_html=True,
        )
        return
    for start in range(0, len(topics), 4):
        cols = st.columns(4, gap="small")
        for offset, col in enumerate(cols):
            i = start + offset
            if i >= len(topics):
                break
            with col:
                with st.container(key=f"topic-{i}"):
                    st.button(f"{topics[i]} \u00d7", key=f"up_rm_{i}", on_click=remove_topic, args=(i,))


def syllabus_card(exam: str) -> None:
    with st.container(key="card-3"):
        st.markdown(form.card_heading(3, "Syllabus for this paper"), unsafe_allow_html=True)
        st.markdown(
            '<div class="card-note">The syllabus changes every semester. Add exactly what is included.</div>',
            unsafe_allow_html=True,
        )
        topic_chips()
        c_in, c_add = st.columns([4, 0.8], gap="small", vertical_alignment="bottom")
        c_in.text_input(
            "Add a topic",
            placeholder="Add a topic, e.g. Hashing",
            label_visibility="collapsed",
            key="up_topic_in",
            on_change=add_topic,
        )
        with c_add:
            with st.container(key="up_topic_add"):
                st.button("Add", key="up_add_btn", on_click=add_topic)
        file = st.file_uploader("Or upload syllabus PDF", type=inputs.FILE_TYPES, key="up_syll_file")
        if file is not None:
            read_syllabus(file, exam)
        if st.session_state["up_syll_msg"]:
            st.caption(st.session_state["up_syll_msg"])
        others = [i for i in st.session_state["up_syll_items"] if i not in st.session_state["up_topics"]]
        if others:
            with st.expander(f"Other items found in the file ({len(others)})"):
                st.multiselect("Add to the syllabus", others, key="up_syll_more", on_change=add_more_topics)
        st.checkbox("Strict syllabus (only questions that match these topics)", key="up_strict")
        st.markdown(
            '<div class="card-note">Syllabus matching compares words, so it is approximate.</div>',
            unsafe_allow_html=True,
        )


# ---------- card 4: timer ----------
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


# ---------- right column ----------
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
    stored = st.session_state["up_upload"]
    extra = stored["result"] if stored and stored["subject"] == subject else None
    try:
        with st.spinner("Reading the archive and building your paper..."):
            paper = get_generator().generate(
                subject,
                exam,
                int(st.session_state["up_year"]),
                model=advanced["model"],
                syllabus_items=list(st.session_state["up_topics"]) or None,
                extra=extra,
                n_questions=advanced["n_questions"] or None,
                total_marks=advanced["total_marks"] or None,
                explore_share=advanced["explore_share"],
                strict_syllabus=bool(st.session_state.get("up_strict", False)),
            )
    except LowConfidenceError as err:
        return str(err)
    except Exception:
        return "Something went wrong while building the paper. Please try again."
    if st.session_state["up_timer_on"]:
        paper["duration_minutes"] = effective_minutes()
    st.session_state["up_exam_timer"] = bool(st.session_state["up_timer_on"])
    st.session_state["up_revealed"] = []
    st.session_state["up_pdf"] = None
    timer.reset()
    st.session_state["up_solutions"] = None
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
        upload_card(st.session_state["up_subject"])
        syllabus_card(st.session_state["up_exam"])
        timer_card()
    with right:
        right_column()


# ---------- predicted paper screen ----------
def exam_label(paper: dict) -> str:
    labels = dict(form.EXAMS)
    return labels.get(paper.get("exam_kind")) or labels.get(st.session_state["up_exam"], "")


def exam_minutes(paper: dict) -> int | None:
    """Minutes for the exam timer (editable on the paper screen), or None when the timer is off."""
    minutes = paper.get("duration_minutes")
    if not (st.session_state["up_exam_timer"] and minutes):
        return None
    return int(st.session_state.get(f"up_exam_min_{int(minutes)}", minutes))


def start_exam() -> None:
    paper = st.session_state["up_paper"]
    chosen = exam_minutes(paper)
    if chosen:
        timer.start(chosen * 60)
    else:
        timer.reset()
    st.session_state["up_screen"] = "exam"


def leave_exam() -> None:
    timer.reset()
    st.session_state["up_screen"] = "paper"


def pdf_bytes(paper: dict) -> bytes:
    if st.session_state["up_pdf"] is None:
        try:
            st.session_state["up_pdf"] = paper_to_pdf(paper)
        except Exception:
            st.session_state["up_pdf"] = b""
    return st.session_state["up_pdf"]


def question_row(index: int, question: dict) -> None:
    redacted = bool(question.get("explore")) and index not in st.session_state["up_revealed"]
    with st.container(key=f"pq-{index}"):
        c_num, c_body = st.columns([0.8, 8], vertical_alignment="top")
        c_num.markdown(
            f'<div class="qn">{theme.esc(predicted.question_label(question, index))}</div>',
            unsafe_allow_html=True,
        )
        with c_body:
            if redacted:
                st.markdown(predicted.redacted_markup(), unsafe_allow_html=True)
                c_tag, c_btn, _ = st.columns([1.6, 1.8, 3], vertical_alignment="center")
                c_tag.markdown('<span class="tag-explore">Less-predicted area</span>', unsafe_allow_html=True)
                with c_btn:
                    st.button("Declassify this question", key=f"declass-{index}", on_click=reveal, args=(index,))
            else:
                st.markdown(predicted.question_markup(question), unsafe_allow_html=True)


def next_move(paper: dict) -> None:
    with st.container(key="next-move"):
        st.markdown('<div class="nm-h">Your next move</div>', unsafe_allow_html=True)
        chosen = exam_minutes(paper)
        label = "Start with timer " + predicted.hhmmss(chosen * 60) if chosen else "Start exam mode"
        with st.container(key="mv-start"):
            st.button(label, key="up_start_exam", on_click=start_exam)
        with st.container(key="mv-solution"):
            st.button("See solution", key="up_see_solution", on_click=go, args=("solved",))
        with st.container(key="mv-upload"):
            st.button("Upload my solution", key="up_upload_solution", on_click=open_results, args=(paper,))
        if chosen:
            st.number_input(
                "Timer minutes (you can change it)", 5, 360, int(paper["duration_minutes"]), 5,
                key=f"up_exam_min_{int(paper['duration_minutes'])}",
            )
        data = pdf_bytes(paper)
        if data:
            name = re.sub(r"[^A-Za-z0-9]+", "_", str(paper.get("title", "paper"))).strip("_") or "paper"
            with st.container(key="mv-dl"):
                st.download_button("Download as PDF", data, file_name=f"{name}.pdf", mime="application/pdf", key="up_pdf_btn")
        else:
            st.caption("PDF export is not available right now.")
    st.markdown(
        '<div class="fine">Questions are real past questions chosen for likely topics. '
        'The squares are a ranking, not an exact chance.</div>',
        unsafe_allow_html=True,
    )


def how_it_was_made(paper: dict) -> None:
    with st.expander("How this paper was made"):
        for line in paper.get("notes") or []:
            st.markdown(f"- {line}")
        st.markdown("- Difficulty is a proxy built from past papers. Syllabus matching compares words and is approximate.")
        fig = predicted.topic_figure(paper.get("topics"))
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Yellow = Likely, blue = Possible, grey = Unlikely. "
                "A red diamond means the topic was already examined earlier this year."
            )


def paper_screen(paper: dict) -> None:
    st.button("\u2190 Change inputs", key="up_back_form", on_click=back_to_form)
    main, side = st.columns([3.1, 1.15], gap="large")
    with main:
        with st.container(key="paper-hero"):
            st.markdown(
                predicted.header_markup(
                    paper.get("subject", ""), exam_label(paper),
                    paper.get("duration_minutes"), paper.get("total_marks"),
                ),
                unsafe_allow_html=True,
            )
            questions = paper.get("questions") or []
            if not questions:
                theme.notice("No questions could be chosen for these inputs. Read the notes below.")
            for i, question in enumerate(questions):
                question_row(i, question)
    with side:
        theme.note("real", "Difficulty", predicted.realised_text(paper), "yellow", "R")
        next_move(paper)
    how_it_was_made(paper)


def exam_question(index: int, question: dict) -> None:
    c_num, c_body = st.columns([0.8, 8], vertical_alignment="top")
    c_num.markdown(
        f'<div class="qn">{theme.esc(predicted.question_label(question, index))}</div>',
        unsafe_allow_html=True,
    )
    marks = question.get("marks")
    chip = ""
    if marks is not None and marks == marks:
        chip = f' <span class="qchip">{float(marks):g} marks</span>'
    text = theme.esc(question.get("text", "")).replace("\n", "<br>")
    c_body.markdown(f'<div class="qt">{text}{chip}</div>', unsafe_allow_html=True)


def exam_screen(paper: dict) -> None:
    """Exam mode: questions with marks only, live clock on the right."""
    st.markdown("<style>.st-key-topnav{display:none !important;}</style>", unsafe_allow_html=True)
    title = f"{paper.get('subject', '')} {exam_label(paper)}".strip()
    c_logo, c_badge, c_exit = st.columns([3, 4, 1.2], vertical_alignment="center")
    c_logo.markdown('<div class="logo">SPILL THE PAPER</div>', unsafe_allow_html=True)
    c_badge.markdown(
        f'<div style="text-align:right"><span class="exam-badge">EXAM MODE {theme.DOT} {theme.esc(title)}</span></div>',
        unsafe_allow_html=True,
    )
    c_exit.button("Exit exam", key="ex_exit", on_click=leave_exam)

    main, side = st.columns([3.1, 1.15], gap="large")
    with main:
        with st.container(key="exam-card"):
            st.markdown(
                '<div class="kicker">Solve on paper, then photograph your answers</div>'
                f'<div class="ph-h">{theme.esc(title)}</div><div class="dash"></div>',
                unsafe_allow_html=True,
            )
            for i, question in enumerate(paper.get("questions") or []):
                with st.container(key=f"eq-{i}"):
                    exam_question(i, question)
    with side:
        if timer.active():
            timer.clock()
        else:
            st.markdown(
                '<div class="st-key-timer-off">Timer is off for this exam. Take your time.</div>',
                unsafe_allow_html=True,
            )
        theme.note("tip", "", "Start with the marks you are sure of. Coding question last.", "yellow", "L")
        with st.container(key="ex-finish-up"):
            st.button("Finish and upload my solution " + theme.ARROW, key="ex_finish_up", on_click=open_results, args=(paper,))
        with st.container(key="ex-finish"):
            st.button("Finish exam", key="ex_finish", on_click=leave_exam)


# ---------- solved paper screen ----------
def solutions_for(paper: dict) -> dict:
    """Archive solutions (instant), computed once per paper and kept in session state."""
    stored = st.session_state["up_solutions"]
    if stored is None:
        try:
            with st.spinner("Looking up solutions in the archive..."):
                items = list(get_solutions().solve_paper(paper, allow_ai=False))
        except Exception:
            items = []
        stored = {"items": items, "pdf": None}
        st.session_state["up_solutions"] = stored
    return stored


def generate_ai(paper: dict, stored: dict, missing: list[int]) -> None:
    """Ask the AI for the questions without an archive solution, one by one with a progress bar."""
    service = get_solutions()
    items = list(stored["items"])
    bar = st.progress(0.0, text="Asking the AI (a few seconds per question)...")
    for step, i in enumerate(missing, 1):
        try:
            items[i] = service.solve(paper["subject"], paper["questions"][i], allow_ai=True)
        except Exception:
            items[i] = dict(items[i], note="The AI solution could not be generated. Please try again.")
        bar.progress(step / len(missing), text=f"AI solution {step} of {len(missing)}")
    bar.empty()
    st.session_state["up_solutions"] = {"items": items, "pdf": None}


def solved_pdf(paper: dict, stored: dict) -> bytes:
    if stored["pdf"] is None:
        try:
            stored["pdf"] = paper_to_pdf(paper, solutions=stored["items"])
        except Exception:
            stored["pdf"] = b""
    return stored["pdf"]


def solution_card(index: int, question: dict, entry: dict) -> None:
    source = entry.get("source", "none")
    label = predicted.question_label(question, index)
    with st.container(key=f"sq-{index}"):
        st.markdown(
            solved.card_header(
                label, question.get("topic_label"), question.get("marks"),
                bool(question.get("leetcode_style")), source,
            ),
            unsafe_allow_html=True,
        )
        with st.expander("Show the question"):
            st.markdown(
                f'<div class="sq-q">{theme.esc(question.get("text", ""))}</div>', unsafe_allow_html=True
            )
        text = str(entry.get("text") or "")
        if source == "none" or not text.strip():
            theme.notice(str(entry.get("note") or "No solution is available for this question yet."))
            return
        for kind, part in solved.split_blocks(text):
            if kind == "code":
                st.markdown(solved.code_markup(part), unsafe_allow_html=True)
            else:
                st.markdown(part)
        if entry.get("note"):
            st.markdown(f'<div class="sq-note">{theme.esc(entry["note"])}</div>', unsafe_allow_html=True)


def solved_screen(paper: dict) -> None:
    stored = solutions_for(paper)
    items = stored["items"]
    questions = paper.get("questions") or []

    c_title, c_back, c_up = st.columns([3.2, 1, 1.2], vertical_alignment="bottom")
    c_title.markdown(
        '<div class="kicker">Declassified</div>'
        '<div class="sp-title">The <span class="hl">solved</span> paper</div>',
        unsafe_allow_html=True,
    )
    c_back.button("\u2190 Back to paper", key="up_back_paper", on_click=go, args=("paper",))
    with c_up:
        with st.container(key="sv-up-top"):
            st.button("Upload my solution", key="sv_up_top", on_click=open_results, args=(paper,))
    st.caption("Archive solutions can be partial, and AI solutions must be checked before you trust them.")

    main, side = st.columns([3.1, 1.15], gap="large")
    with main:
        if not items:
            theme.notice("Solutions could not be loaded right now. Go back to the paper and try again.")
        for i, entry in enumerate(items):
            if i < len(questions):
                solution_card(i, questions[i], entry)
    with side:
        for i, entry in enumerate(items):
            scheme = entry.get("marking_scheme")
            if scheme and i < len(questions):
                label = predicted.question_label(questions[i], i)
                theme.note(f"mark-{i}", f"Marking scheme {label}", str(scheme), "yellow", "R")
        theme.note(
            "slipped", "", "Solved it yourself first? Upload a photo and we will mark where you slipped.", "pink", "L"
        )
        with st.container(key="sv-up"):
            st.button("Upload my solution " + theme.ARROW, key="sv_up", on_click=open_results, args=(paper,))

        missing = [i for i, e in enumerate(items) if e.get("source") == "none"]
        if get_solutions().ai_available():
            if missing and st.button("Generate missing solutions with AI", key="up_gen_ai"):
                generate_ai(paper, stored, missing)
                st.rerun()
        else:
            st.caption("AI solutions are switched off (no API key set).")

        if items:
            data = solved_pdf(paper, stored)
            if data:
                with st.container(key="sv-pdf"):
                    st.download_button(
                        "Download solved paper as PDF", data, file_name="solved_paper.pdf",
                        mime="application/pdf", key="up_solved_pdf_btn",
                    )


_paper = st.session_state["up_paper"]
_screen = st.session_state["up_screen"]
if _paper and _screen == "paper":
    paper_screen(_paper)
elif _paper and _screen == "exam":
    exam_screen(_paper)
elif _paper and _screen == "solved":
    solved_screen(_paper)
else:
    form_screen()