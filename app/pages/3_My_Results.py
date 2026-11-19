"""My results: upload a solution, check what was read, mark it, see the weak areas,
save it to the history and generate a personalised paper."""

import bootstrap  # noqa: F401  (puts the project root on sys.path)

import datetime
import time

import streamlit as st
import theme
from common import get_generator, get_llm, get_results, get_solutions
from components import results_ui as ui
from components import solved
from theme import DOT, esc

from src.inference.errors import LowConfidenceError
from src.inference.grading import grade_paper, manual_grades, override, read_answers, split_answers
from src.inference.personalised import generate_personalised
from src.inference.results import analyse
from src.inference.solution_jobs import SolutionJob

ss = st.session_state
theme.setup("My results", active="results")
ui.inject_css()
solved.inject_css()

APPROX = "AI marking is approximate. You can change any mark."
LOCAL = "Saved on this computer only."


# ------------------------------------------------------------------- navigation
def go(screen: str) -> None:
    ss["res_screen"] = screen


def clear_score_keys() -> None:
    for key in [k for k in ss if str(k).startswith("res_score_")]:
        del ss[key]


def set_grades(grades: list[dict]) -> None:
    clear_score_keys()
    ss["res_grades"] = grades


def prefill_answers(answers: dict) -> None:
    for q in ss["res_paper"]["questions"]:
        ss[f"res_ans_{q['number']}"] = answers.get(q["number"], "")


def to_review() -> None:
    """Back from the marking screen: keep the (possibly edited) answers."""
    prefill_answers(ss.get("res_answers", {}))
    go("review")


def to_marking() -> None:
    ss["res_answers"] = {q["number"]: (ss.get(f"res_ans_{q['number']}") or "").strip()
                         for q in ss["res_paper"]["questions"]}
    for key in ("res_grades", "res_analysis", "res_saved_id"):
        ss.pop(key, None)
    clear_score_keys()
    go("marking")


def on_score(number: int) -> None:
    """The student typed a mark for one question."""
    value = ss.get(f"res_score_{number}")
    if value is not None and ss.get("res_grades"):
        ss["res_grades"] = override(ss["res_grades"], number, value)


def title_row(kicker: str, plain: str, highlighted: str) -> None:
    theme.title(kicker, plain, highlighted)
    paper = ss.get("res_paper")
    if paper:
        st.markdown(f'<div class="rs-sub">{esc(paper["title"])}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------- home
def delete_ask(attempt_id: str) -> None:
    ss["res_confirm"] = attempt_id


def delete_no() -> None:
    ss.pop("res_confirm", None)


def delete_yes(attempt_id: str) -> None:
    get_results().delete(attempt_id)
    ss.pop("res_confirm", None)


def home() -> None:
    theme.title("Evidence drop", "My", "results")
    paper = ss.get("res_paper")
    if paper:
        with st.container(key="rs-taped"):
            st.markdown(f'<div class="rs-sub">Just attempted</div><div class="rs-t">Mark the paper you just '
                        f'attempted: {esc(paper["title"])}</div>', unsafe_allow_html=True)
            st.button("Upload my solution", key="rs-go-home", on_click=go, args=("upload",))

    store = get_results()
    attempts = store.attempts()
    if not attempts:
        with st.container(key="rs-card-empty"):
            st.markdown('<div class="rs-t">No marked papers yet.</div><div class="rs-text">Predict a paper, solve it, '
                        'then come back to upload your answers and see your weak areas.</div>',
                        unsafe_allow_html=True)
            c1, c2, _ = st.columns([1, 1, 2])
            c1.page_link("pages/2_Upcoming_Paper.py", label="Predict a paper")
            c2.page_link("pages/1_Past_Papers.py", label="Browse past papers")
        return

    subjects = sorted({a["subject"] for a in attempts})
    st.markdown('<div class="rs-t">Your progress</div>', unsafe_allow_html=True)
    subject = st.selectbox("Subject", subjects, key="res_subject")
    chosen = store.attempts(subject)
    fig = ui.history_figure(chosen)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    weak_fig = ui.weakness_figure(store.weakness(subject))
    if weak_fig is not None:
        st.markdown('<div class="rs-t">Weak areas over your last attempts</div>'
                    '<div class="rs-note">Newer attempts count more. A topic seen in one attempt only is a hint.</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(weak_fig, use_container_width=True)

    st.markdown('<div class="rs-t">History</div>', unsafe_allow_html=True)
    for a in attempts:
        with st.container(key=f"rs-card-{a['id']}"):
            st.markdown(ui.attempt_markup(a), unsafe_allow_html=True)
            if ss.get("res_confirm") == a["id"]:
                st.markdown('<div class="rs-note">Delete this result? It cannot be undone.</div>',
                            unsafe_allow_html=True)
                c1, c2, _ = st.columns([1, 1, 3])
                c1.button("Yes, delete", key=f"rs-danger-yes-{a['id']}", on_click=delete_yes, args=(a["id"],))
                c2.button("Keep it", key=f"res_keep_{a['id']}", on_click=delete_no)
            else:
                st.button("Delete", key=f"res_del_{a['id']}", on_click=delete_ask, args=(a["id"],))
    st.markdown(f'<div class="rs-note">{LOCAL}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------- upload
def upload() -> None:
    paper, llm = ss["res_paper"], get_llm()
    title_row("Evidence drop", "Upload your", "solution")
    theme.notice("Photograph every page of your answers. Write Q1, Q2 ... next to each answer. "
                 "Photos and answers stay on this computer.")
    files = st.file_uploader("Photos or files of your answers", key="res_files", accept_multiple_files=True,
                             type=["jpg", "jpeg", "png", "pdf", "docx", "txt"])
    typed = st.text_area("Or type your answers", key="res_typed", height=200,
                         placeholder="Q1 your answer ...\nQ2 your answer ...")
    has_photo = any(f.name.lower().endswith((".jpg", ".jpeg", ".png")) for f in files or [])
    if has_photo and not llm.supports_vision:
        theme.notice("Reading handwriting from photos needs an AI key (GEMINI_API_KEY). Without one, only typed "
                     "text and clean printed scans can be read. You can type your answers instead.")
    if not llm.available:
        st.caption("No AI key is set: typed answers and marking yourself still work.")

    c1, c2, _ = st.columns([1.4, 1, 2.6])
    read = c1.button("Read my answers", key="rs-go-read")
    c2.button("Back", key="res_back_upload", on_click=go, args=("home",))
    if not read:
        return
    if not files and not (typed or "").strip():
        theme.notice("Add at least one photo or file, or type your answers first.")
        return
    numbers = [q["number"] for q in paper["questions"]]
    try:
        with st.spinner("Reading your answers... photos can take 10 to 30 seconds."):
            sheet = read_answers(paper, ui.collect_files(files), llm=llm)
            answers, rest = split_answers(typed or "", numbers)
            sheet = ui.combine_sheet(sheet, answers, rest, numbers)
    except Exception:  # never show a traceback to the student
        theme.notice("Something went wrong while reading your answers. Try again, or type them instead.")
        return
    ss["res_sheet"] = sheet
    prefill_answers(sheet.answers)
    go("review")
    st.rerun()


# ----------------------------------------------------------------------- review
def review() -> None:
    paper, sheet = ss["res_paper"], ss["res_sheet"]
    title_row("Step 2 of 4", "Check what we", "read")
    st.markdown(f'<div class="rs-text">How it was read: {esc(ui.METHOD_TEXT.get(sheet.method, sheet.method))}. '
                'AI reading of handwriting is approximate: fix any text below.</div>', unsafe_allow_html=True)
    for line in sheet.warnings:
        theme.notice(line)
    if sheet.unassigned:
        theme.notice("We found text that could not be tied to a question. Copy the right parts into the "
                     "answers below.")
        st.text_area("Text we could not place", value=sheet.unassigned, key="res_unassigned", height=140)
    for q in paper["questions"]:
        n = q["number"]
        with st.container(key=f"rs-card-q{n}"):
            label = f"Q{n} {DOT} {q['topic_label']}" if q.get("topic_label") else f"Q{n}"
            st.markdown(f'<div class="rs-t">{esc(label)}</div>', unsafe_allow_html=True)
            with st.expander("Show the question"):
                st.markdown(f'<div class="rs-ans">{esc(q.get("text", ""))}</div>', unsafe_allow_html=True)
            st.text_area(f"Your answer to Q{n}", key=f"res_ans_{n}", height=150)
    c1, c2, _ = st.columns([1.4, 1, 2.6])
    c1.button("Mark my answers", key="rs-go-mark", on_click=to_marking)
    c2.button("Back", key="res_back_review", on_click=go, args=("upload",))


# ---------------------------------------------------------------------- marking
def solution_job(paper: dict) -> SolutionJob:
    """The solutions of the paper being marked. When it is the paper whose solutions were already opened on
    the 'See solution' page, that same job is reused (so the AI solutions written there show up here too);
    otherwise one is started: archive solutions at once, the missing ones written by the AI in the background."""
    stored = ss.get("up_solutions")
    if stored and stored.get("paper") is paper:
        return stored["job"]
    mine = ss.get("res_job")
    if mine and mine["paper"] is paper:
        return mine["job"]
    job = SolutionJob(get_solutions(), paper)
    ss["res_job"] = {"job": job, "paper": paper}
    return job


def start_manual() -> None:
    paper, answers = ss["res_paper"], ss["res_answers"]
    set_grades(grade_paper(paper, answers, solutions=solution_job(paper).snapshot(), llm=get_llm(), allow_ai=False))


def run_ai_marking(paper: dict, answers: dict, job: SolutionJob) -> None:
    if job.pending:      # the AI marks against the reference solutions, so let the missing ones finish first
        with st.spinner("Waiting for the reference solutions the AI is still writing..."):
            deadline = time.time() + 90
            while job.pending and time.time() < deadline:
                time.sleep(1)
    solutions = job.snapshot()
    grades: list[dict] = []
    progress = st.progress(0.0, text="Marking with AI...")
    for i, q in enumerate(paper["questions"], 1):
        grades += grade_paper({"questions": [q]}, answers, solutions=solutions, llm=get_llm())
        progress.progress(i / len(paper["questions"]), text=f"Marked {i} of {len(paper['questions'])}")
    set_grades(grades)


def reference_markup(solution: dict | None) -> None:
    text = (solution or {}).get("text") or ""
    if not text:      # neither the archive nor the AI has a solution: say why (or that the AI is still writing it)
        note = (solution or {}).get("note") or "No reference solution is available for this question."
        st.markdown(f'<div class="rs-note">{esc(note)}</div>', unsafe_allow_html=True)
        return
    if solution.get("source") == "ai":
        st.markdown('<span class="rs-badge weak">AI-generated: verify</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="rs-badge">From the archive</span>', unsafe_allow_html=True)
    for kind, part in solved.split_blocks(text):
        if kind == "code":
            st.markdown(solved.code_markup(part), unsafe_allow_html=True)
        else:
            st.markdown(part)


def reference_block(job: SolutionJob, number: int, polling: bool) -> None:
    """The reference solution of one question. While the AI is still writing it, it redraws every 2 seconds."""
    solution = next((e for e in job.snapshot() if e["number"] == number), None)
    reference_markup(solution)
    if polling and not job.pending:
        st.rerun()      # everything is written: redraw the page once so polling stops


def grade_card(q: dict, g: dict, job: SolutionJob) -> None:
    n = g["number"]
    with st.container(key=f"rs-card-g{n}"):
        st.markdown(ui.grade_header(f"Q{n}", g), unsafe_allow_html=True)
        line = ui.status_line(g)
        if line:
            st.markdown(f'<div class="rs-text">{esc(line)}</div>', unsafe_allow_html=True)
        st.markdown(ui.missing_markup(g.get("missing") or []), unsafe_allow_html=True)
        with st.expander("Show the question"):
            st.markdown(f'<div class="rs-ans">{esc(q.get("text", ""))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="rs-note">Your answer:</div>' + ui.answer_markup(g.get("answer") or ""),
                    unsafe_allow_html=True)
        with st.expander("Reference solution"):
            if job.pending:
                st.fragment(run_every=2)(reference_block)(job, n, True)
            else:
                reference_block(job, n, False)
        if not g.get("marks_known", True):
            st.caption("Marks not stated for this question, counted as 1.")
        st.number_input(f"Your mark for Q{n} (out of {g['marks']:g})", min_value=0.0, max_value=float(g["marks"]),
                        step=0.5, value=None if g["score"] is None else float(g["score"]),
                        key=f"res_score_{n}", on_change=on_score, args=(n,))


def marking() -> None:
    paper, answers, llm = ss["res_paper"], ss["res_answers"], get_llm()
    title_row("Step 3 of 4", "Mark your", "answers")
    job = solution_job(paper)

    if ss.get("res_grades") is None:
        if llm.available:
            with st.container(key="rs-card-how"):
                st.markdown('<div class="rs-t">How do you want it marked?</div><div class="rs-text">The AI marks one '
                            'question at a time (2 to 5 seconds each). You can change every mark afterwards.</div>',
                            unsafe_allow_html=True)
                c1, c2, c3, _ = st.columns([1.3, 1.3, 1, 1.4])
                ai = c1.button("Mark with AI", key="rs-go-ai")
                c2.button("I will mark myself", key="res_manual", on_click=start_manual)
                c3.button("Back", key="res_back_marking0", on_click=to_review)
            if ai:
                run_ai_marking(paper, answers, job)
                st.rerun()
            return
        start_manual()  # no key: the student marks against the reference solutions
        st.caption("No AI key is set, so you mark yourself. Compare each answer with the reference solution.")

    grades = ss["res_grades"]
    theme.notice(APPROX)
    failed = [g["number"] for g in grades if g["status"] == "not_graded" and g.get("answer")]
    if failed and llm.available:
        theme.notice("The AI could not mark question(s) " + ", ".join(map(str, failed)) +
                     " (the service may be busy). Mark them yourself below, or go back and try again.")
    by_number = {q["number"]: q for q in paper["questions"]}
    for g in grades:
        grade_card(by_number[g["number"]], g, job)

    c1, c2, _ = st.columns([1.5, 1.3, 2.2])
    if c1.button("Show my weak areas", key="rs-go-report", disabled=not ui.can_report(grades)):
        try:
            ss["res_analysis"] = analyse(paper, grades)
        except ValueError:
            theme.notice("Mark at least one answer first.")
            return
        go("report")
        st.rerun()
    c2.button("Back to my answers", key="res_back_marking", on_click=to_review)


# ----------------------------------------------------------------------- report
def save_attempt(paper: dict, grades: list[dict], analysis: dict) -> None:
    try:
        ss["res_saved_id"] = get_results().add(paper, grades, analysis)
    except OSError:
        theme.notice("The result could not be saved on this computer.")


def report() -> None:
    paper, grades = ss["res_paper"], ss["res_grades"]
    title_row("Marked", "Your weak", "areas")
    try:
        analysis = analyse(paper, grades)
    except ValueError:
        theme.notice("Nothing is marked yet. Go back and mark at least one answer.")
        st.button("Back to the questions", key="res_back_empty", on_click=go, args=("marking",))
        return
    ss["res_analysis"] = analysis
    o = analysis["overall"]
    with st.container(key="rs-card-score"):
        st.markdown(f'<div class="rs-h"><div class="rs-big">{o["score"]:g}<small> / {o["total"]:g}</small></div>'
                    f'<div class="stamp">MARKED</div></div>'
                    f'<div class="rs-badges"><span class="rs-badge {ui.ratio_band(o["percent"], 100)}">'
                    f'{o["percent"]:.0f}%</span><span class="rs-badge">{o["counted"]} of {o["questions"]} '
                    f'questions counted</span></div>', unsafe_allow_html=True)
        for line in analysis["summary"]:
            st.markdown(f'<div class="rs-text">{esc(line)}</div>', unsafe_allow_html=True)
    fig = ui.topic_figure(analysis)
    if fig is not None:
        with st.container(key="rs-card-chart"):
            st.markdown('<div class="rs-t">Marks by topic</div><div class="rs-note">Pink = weak (under 50%), '
                        'yellow = ok, blue = strong (75% or more).</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
    with st.container(key="rs-card-topics"):
        st.markdown('<div class="rs-t">Topics</div>' + ui.table_markup(
            ["Topic", "Marks", "Percent", "Band", "Questions"], ui.topic_rows(analysis)), unsafe_allow_html=True)
        st.markdown('<div class="rs-note">A topic based on one question is a hint, not a fact.</div>',
                    unsafe_allow_html=True)
    cols = st.columns(2)
    for col, key, name in ((cols[0], "by_difficulty", "Difficulty"), (cols[1], "by_qtype", "Question type")):
        if analysis[key]:
            with col, st.container(key=f"rs-card-{key}"):
                st.markdown(f'<div class="rs-t">By {name.lower()}</div>' + ui.table_markup(
                    [name, "Marks", "Percent", "Questions"], ui.group_rows(analysis[key])), unsafe_allow_html=True)
    if analysis["not_counted"]:
        theme.notice("Not counted (not marked or unreadable): question(s) "
                     + ", ".join(map(str, analysis["not_counted"])) + ".")

    saved = ss.get("res_saved_id") is not None
    c1, c2, c3, _ = st.columns([1.3, 1.6, 1.3, 0.8])
    if c1.button("Saved" if saved else "Save to My results", key="rs-go-save", disabled=saved):
        save_attempt(paper, grades, analysis)
        st.rerun()
    c2.button("Generate a personalised paper", key="rs-go-personal", on_click=go, args=("personalised",))
    c3.button("Back to the questions", key="res_back_report", on_click=go, args=("marking",))
    st.markdown(f'<div class="rs-note">{LOCAL} {APPROX}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------- personalised
def personalised() -> None:
    paper = ss["res_paper"]
    title_row("Made for you", "Personalised", "paper")
    default_exam = ui.exam_for(paper)
    if default_exam is None:
        theme.notice("A personalised paper is not available for this exam type.")
        st.button("Back", key="res_back_personal0", on_click=go, args=("report",))
        return
    analysis = ss.get("res_analysis") or analyse(paper, ss["res_grades"])
    st.markdown('<div class="rs-text">The questions are real past questions chosen for your weak topics. About half '
                'target your weak areas, the rest follow the prediction, and a few less-predicted areas stay in.</div>',
                unsafe_allow_html=True)
    with st.container(key="rs-card-options"):
        exams = list(ui.EXAM_LABELS)
        c1, c2, c3 = st.columns(3)
        exam = c1.selectbox("Exam", exams, index=exams.index(default_exam), format_func=ui.EXAM_LABELS.get,
                            key="res_exam")
        this_year = datetime.date.today().year
        year = c2.number_input("Year", min_value=2000, max_value=this_year + 2, value=this_year, step=1,
                               key="res_year")
        size = c3.selectbox("Number of questions", ["Auto"] + list(range(3, 9)), key="res_n")
        st.caption("More questions give a better picture of your weak areas.")
        b1, b2, _ = st.columns([1.3, 1, 3])
        go_btn = b1.button("Generate", key="rs-go-generate")
        b2.button("Back", key="res_back_personal", on_click=go, args=("report",))
    if not go_btn:
        return
    try:
        with st.spinner("Building your personalised paper..."):
            p = generate_personalised(get_generator(), paper["subject"], exam, int(year), analysis=analysis,
                                      store=get_results(), source_paper=paper,
                                      n_questions=None if size == "Auto" else int(size))
    except LowConfidenceError as e:
        theme.notice(f"A personalised paper is not available for this subject: {e}")
    except ValueError as e:
        theme.notice(f"A personalised paper could not be made: {e}")
    except Exception:
        theme.notice("Something went wrong while building the paper. Please try again.")
    else:
        ss.update({"up_paper": p, "up_screen": "paper", "up_revealed": [], "up_pdf": None,
                   "up_solutions": None, "up_exam_timer": True, "up_subject": p["subject"]})
        st.switch_page("pages/2_Upcoming_Paper.py")


# ---------------------------------------------------------------------- routing
NEEDS = {"upload": (), "review": ("res_sheet",), "marking": ("res_answers",),
         "report": ("res_grades",), "personalised": ("res_grades",)}
SCREENS = {"home": home, "upload": upload, "review": review, "marking": marking,
           "report": report, "personalised": personalised}

screen = ss.get("res_screen", "home")
if screen not in SCREENS or (screen != "home" and not ss.get("res_paper")):
    screen = "home"
elif any(ss.get(k) is None for k in NEEDS.get(screen, ())):
    screen = "upload"
ss["res_screen"] = screen
SCREENS[screen]()