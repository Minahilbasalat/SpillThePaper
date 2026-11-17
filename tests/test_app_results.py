"""Tests for the My results page (upload, review, marking, weak-area report, history, personalised paper).

Helper tests need no data. Page tests use the real runtime bundle (models/ and data/) and are skipped
when it is missing. No test calls a real AI: the API keys are removed and the page gets a client
without a key. File uploads cannot be driven by AppTest, so the upload helpers are tested directly."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for path in (str(ROOT), str(APP)):
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.inference.grading import AnswerSheet, manual_grades  # noqa: E402
from src.inference.results import ResultsStore, analyse  # noqa: E402

HAS_BUNDLE = (ROOT / "models").is_dir() and (ROOT / "data").is_dir()
needs_bundle = pytest.mark.skipif(not HAS_BUNDLE, reason="runtime bundle (models/, data/) is missing")
TIMEOUT = 180
KEYS = ("GEMINI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "ANTHROPIC_API_KEY")


def small_paper(n: int = 3) -> dict:
    """A small hand-made paper (no bundle needed)."""
    qs = [{"number": i, "text": f"Question {i} text", "marks": 10.0, "topic_id": i, "topic_label": f"topic {i}",
           "difficulty": "medium", "qtype": "problem_solving", "solution": f"Solution {i}"} for i in range(1, n + 1)]
    return {"title": "Predicted Final 2026 - Test Subject", "subject": "Test Subject", "exam_kind": "final",
            "year": 2026, "source": "predicted", "duration_minutes": 180, "total_marks": 10.0 * n,
            "questions": qs, "notes": []}


def marked(paper: dict, scores: dict) -> tuple[list, dict]:
    grades = manual_grades(paper, scores, {q["number"]: f"my answer {q['number']}" for q in paper["questions"]})
    return grades, analyse(paper, grades)


# ------------------------------------------------------------------ helper tests
def test_open_results_sets_the_session_keys(monkeypatch):
    jumps = []
    monkeypatch.setattr(st, "switch_page", lambda page: jumps.append(page))

    def script():
        import bootstrap  # noqa: F401
        import streamlit as st
        from results_state import open_results

        if "res_paper" not in st.session_state:  # only before the click: the script reruns after it
            for k in ("res_sheet", "res_grades", "res_analysis", "res_saved_id", "res_personalised"):
                st.session_state[k] = "old"
        st.button("go", key="go", on_click=open_results, args=({"title": "P", "questions": []},))

    at = AppTest.from_function(script, default_timeout=60)
    at.run()
    at.button(key="go").click().run()
    assert at.session_state["res_paper"]["title"] == "P"
    assert at.session_state["res_screen"] == "upload"
    for k in ("res_sheet", "res_grades", "res_analysis", "res_saved_id", "res_personalised"):
        assert k not in at.session_state
    assert jumps == ["pages/3_My_Results.py"]


def test_ratio_band_and_can_report():
    from components import results_ui as ui

    assert ui.ratio_band(None, 10) == "none"
    assert ui.ratio_band(4, 10) == "weak"
    assert ui.ratio_band(5, 10) == "ok"
    assert ui.ratio_band(7.5, 10) == "strong"
    paper = small_paper()
    assert not ui.can_report(manual_grades(paper, {}))
    assert ui.can_report(manual_grades(paper, {1: 5}))
    assert not ui.can_report(None)


def test_status_line_texts():
    from components import results_ui as ui

    assert ui.status_line({"status": "unanswered", "feedback": "x"}) == "No answer found (counts 0)"
    assert ui.status_line({"status": "unreadable", "feedback": "x"}) == "Could not be read (not counted)"
    assert ui.status_line({"status": "not_graded", "feedback": ""}) == "Not marked yet"
    assert ui.status_line({"status": "graded_ai", "feedback": "Good"}) == "Good"


def test_combine_sheet_adds_typed_answers():
    from components import results_ui as ui

    sheet = AnswerSheet(answers={1: "from photo"}, method="vision")
    ui.combine_sheet(sheet, {1: "typed", 2: "second"}, "", [1, 2, 3])
    assert sheet.answers == {1: "from photo\ntyped", 2: "second"}
    assert sheet.method == "mixed"
    assert any("No answer found for question(s) 3" in w for w in sheet.warnings)

    sheet = AnswerSheet()
    ui.combine_sheet(sheet, {}, "some loose text", [1, 2])
    assert sheet.unassigned == "some loose text" and sheet.warnings and sheet.method == "text"

    sheet = AnswerSheet()
    ui.combine_sheet(sheet, {}, "only answer", [1])
    assert sheet.answers == {1: "only answer"}


def test_collect_files():
    from components import results_ui as ui

    class Fake:
        name = "a.txt"

        def getvalue(self):
            return b"Q1 hi"

    assert ui.collect_files([Fake()]) == [{"name": "a.txt", "data": b"Q1 hi"}]
    assert ui.collect_files(None) == []


def test_report_helpers_use_the_analysis():
    from components import results_ui as ui

    paper = small_paper()
    _, analysis = marked(paper, {1: 0, 2: 5, 3: 10})
    rows = ui.topic_rows(analysis)
    assert rows[0][0].startswith("topic 1") and "one question" in rows[0][0]
    assert ui.group_rows(analysis["by_difficulty"])[0][0] == "medium"
    fig = ui.topic_figure(analysis)
    assert fig is not None and list(fig.data[0].y)[-1] == "topic 1"  # the weakest topic is drawn on top
    assert "topic 1" in ui.table_markup(["Topic"], [["topic 1"]])


def test_history_helpers():
    from components import results_ui as ui

    paper = small_paper()
    grades, analysis = marked(paper, {1: 0, 2: 5, 3: 10})
    attempt = {"id": "a", "time": "2026-11-17T01:30:00+00:00", "title": "T <b>", "overall": analysis["overall"],
               "topics": analysis["topics"], "personalised": True}
    assert ui.weakest_labels(attempt) == ["topic 1"]
    markup = ui.attempt_markup(attempt)
    assert "personalised" in markup and "<b>" not in markup  # backend text is escaped
    assert "2026" in ui.date_text(attempt["time"])
    assert ui.date_text("not a date") == "not a date"
    assert ui.history_figure([]) is None and ui.history_figure([attempt, attempt]) is not None
    assert ui.weakness_figure({}) is None


def test_results_store_history(tmp_path):
    store = ResultsStore(tmp_path / "r.json")
    paper = small_paper()
    grades, analysis = marked(paper, {1: 0, 2: 5, 3: 10})
    first = store.add(paper, grades, analysis)
    second = store.add(paper, grades, analysis)
    assert [a["id"] for a in store.attempts()] == [second, first]
    assert store.weakness("Test Subject")[1]["weakness"] == 1.0
    assert store.delete(first) and not store.delete(first)


# --------------------------------------------------------------------- page tests
@pytest.fixture
def env(monkeypatch, tmp_path):
    """No AI key, a results store in a temporary folder, switch_page recorded."""
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    import common
    from src.inference.llm import LLMClient

    store = ResultsStore(tmp_path / "r.json")
    monkeypatch.setattr(common, "get_results", lambda: store)
    monkeypatch.setattr(common, "get_llm", lambda: LLMClient())
    jumps = []
    monkeypatch.setattr(st, "switch_page", lambda page: jumps.append(page))
    return {"store": store, "jumps": jumps}


@pytest.fixture(scope="module")
def real_paper():
    if not HAS_BUNDLE:
        pytest.skip("runtime bundle is missing")
    from src.inference.generator import PaperGenerator
    from src.inference.library import Library
    from src.inference.predictor import Predictor

    predictor = Predictor()
    gen = PaperGenerator(predictor, Library())
    subject = "Data Structures" if "Data Structures" in predictor.subjects() else predictor.subjects()[0]
    return gen.generate(subject, "final", 2026, seed=1)


def run_page(**state) -> AppTest:
    """Open the page the way a visitor does (through Home.py), with some session state preloaded."""
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=TIMEOUT)
    at.run()
    for k, v in state.items():
        at.session_state[k] = v
    at.switch_page("pages/3_My_Results.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def all_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


@needs_bundle
def test_home_empty_shows_friendly_card(env):
    at = run_page()
    assert "No marked papers yet" in all_text(at)
    assert "Coming soon" not in all_text(at)


@needs_bundle
def test_home_with_history_and_delete(env):
    paper = small_paper()
    grades, analysis = marked(paper, {1: 0, 2: 5, 3: 10})
    attempt_id = env["store"].add(paper, grades, analysis)
    at = run_page()
    assert "Predicted Final 2026" in all_text(at) and "topic 1" in all_text(at)
    at.button(key=f"res_del_{attempt_id}").click().run()
    assert at.session_state["res_confirm"] == attempt_id
    at.button(key=f"rs-danger-yes-{attempt_id}").click().run()
    assert env["store"].attempts() == []
    assert not at.exception, [e.value for e in at.exception]


@needs_bundle
def test_home_offers_the_paper_just_attempted(env):
    at = run_page(res_paper=small_paper(), res_screen="home")
    assert "Mark the paper you just attempted" in all_text(at)
    at.button(key="rs-go-home").click().run()
    assert at.session_state["res_screen"] == "upload"


@needs_bundle
def test_upload_screen_without_key_says_so(env):
    at = run_page(res_paper=small_paper(), res_screen="upload")
    assert any("No AI key" in c.value for c in at.caption)
    at.button(key="rs-go-read").click().run()  # nothing added: friendly message, no crash
    assert at.session_state["res_screen"] == "upload"
    assert "Add at least one photo" in all_text(at)


@needs_bundle
def test_typed_answers_are_read_and_reviewed(env):
    at = run_page(res_paper=small_paper(), res_screen="upload")
    at.text_area(key="res_typed").set_value("Q1 first answer\nQ2 second answer").run()
    at.button(key="rs-go-read").click().run()
    assert at.session_state["res_screen"] == "review"
    assert at.text_area(key="res_ans_1").value == "first answer"
    assert at.text_area(key="res_ans_2").value == "second answer"
    assert at.text_area(key="res_ans_3").value == ""
    assert "typed text" in all_text(at)
    assert any("No answer found for question" in n for n in [m.value for m in at.markdown])
    at.text_area(key="res_ans_3").set_value("fixed by hand").run()
    at.button(key="rs-go-mark").click().run()
    assert at.session_state["res_answers"][3] == "fixed by hand"
    assert at.session_state["res_screen"] == "marking"


@needs_bundle
def test_marking_without_key_is_manual_and_overridable(env):
    paper = small_paper()
    answers = {1: "a1", 2: "a2", 3: ""}
    at = run_page(res_paper=paper, res_screen="marking", res_answers=answers)
    assert "AI marking is approximate" in all_text(at)
    assert len(at.number_input) == 3
    assert at.button(key="rs-go-report").disabled is False  # the unanswered question already counts 0
    at.number_input(key="res_score_1").set_value(7.5).run()
    g = {x["number"]: x for x in at.session_state["res_grades"]}
    assert g[1]["score"] == 7.5 and g[1]["status"] == "manual"
    assert g[3]["status"] == "unanswered"
    at.button(key="rs-go-report").click().run()
    assert at.session_state["res_screen"] == "report"


@needs_bundle
def test_report_shows_weak_topics_and_saves_once(env):
    paper = small_paper()
    grades, analysis = marked(paper, {1: 0, 2: 5, 3: 10})
    at = run_page(res_paper=paper, res_screen="report", res_grades=grades)
    text = all_text(at)
    assert "15 / 30" in html.unescape(text).replace("<small>", "").replace("</small>", "")
    assert html.escape("topic 1") in text and "Weak areas" in text
    at.button(key="rs-go-save").click().run()
    assert len(env["store"].attempts()) == 1
    assert at.session_state["res_saved_id"]
    assert at.button(key="rs-go-save").disabled is True
    assert not at.exception, [e.value for e in at.exception]


@needs_bundle
def test_report_with_nothing_marked_is_friendly(env):
    paper = small_paper()
    at = run_page(res_paper=paper, res_screen="report", res_grades=manual_grades(paper, {}))
    assert "Nothing is marked yet" in all_text(at) or any("Nothing is marked yet" in m.value for m in at.markdown)


@needs_bundle
def test_missing_state_falls_back_without_error(env):
    for screen in ("review", "marking", "report", "personalised"):
        at = run_page(res_paper=small_paper(), res_screen=screen)
        assert at.session_state["res_screen"] == "upload"
    at = run_page(res_screen="report")  # no paper at all
    assert at.session_state["res_screen"] == "home"


@needs_bundle
def test_personalised_screen_for_unsupported_exam(env):
    paper = small_paper()
    paper["exam_kind"] = "other"
    grades, _ = marked(paper, {1: 0, 2: 5, 3: 10})
    at = run_page(res_paper=paper, res_screen="personalised", res_grades=grades)
    assert any("not available for this exam type" in m.value for m in at.markdown)


@needs_bundle
def test_personalised_paper_is_handed_to_the_paper_screen(env, real_paper):
    grades, analysis = marked(real_paper, {q["number"]: 0 for q in real_paper["questions"][:2]})
    at = run_page(res_paper=real_paper, res_screen="personalised", res_grades=grades, res_analysis=analysis)
    assert at.selectbox(key="res_exam").value == "final"
    at.button(key="rs-go-generate").click().run()
    assert not at.exception, [e.value for e in at.exception]
    if env["jumps"] == []:
        pytest.skip("the generator declined this subject (low confidence); the message was shown")
    assert env["jumps"] == ["pages/2_Upcoming_Paper.py"]
    p = at.session_state["up_paper"]
    assert p["personalised"] is True and at.session_state["up_screen"] == "paper"
    assert at.session_state["up_exam_timer"] is True and at.session_state["up_subject"] == p["subject"]