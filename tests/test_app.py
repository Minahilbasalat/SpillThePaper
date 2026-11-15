"""Smoke tests for the Streamlit UI (features 1-4).

Helper tests need no data. Page tests run every page with the real runtime bundle
(models/ and data/) and are skipped when the bundle is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for path in (str(ROOT), str(APP)):
    if path not in sys.path:
        sys.path.insert(0, path)

from streamlit.testing.v1 import AppTest  # noqa: E402

HAS_BUNDLE = (ROOT / "models").is_dir() and (ROOT / "data").is_dir()
needs_bundle = pytest.mark.skipif(not HAS_BUNDLE, reason="runtime bundle (models/, data/) is missing")

PAGES = ["Home.py", "pages/1_Past_Papers.py", "pages/2_Upcoming_Paper.py", "pages/3_My_Results.py"]
TIMEOUT = 180


# ---------- pure helpers (no data needed) ----------
def test_exam_for_kind():
    import state

    assert state.exam_for_kind("final") == "final"
    assert state.exam_for_kind("mid1") == "mid1"
    assert state.exam_for_kind("quiz3") == "quiz"
    assert state.exam_for_kind("other") is None


def test_difficulty_percentages():
    from components import predict_form as form

    assert form.difficulty_percentages(None) is None
    assert form.difficulty_percentages({"easy": 0, "medium": 0, "hard": 0}) is None
    pct = form.difficulty_percentages({"easy": 0.3, "medium": 0.45, "hard": 0.25})
    assert pct == [30, 45, 25]
    assert sum(form.difficulty_percentages({"easy": 1, "medium": 1, "hard": 1})) == 100


def test_upload_guesses():
    from components import inputs

    assert inputs.guess_kind("Quiz3_scan.jpg") == "quiz"
    assert inputs.guess_kind("CN_Final_2024.pdf") == "final"
    assert inputs.guess_kind("paper.pdf") == "mid"
    assert inputs.guess_year("Mid1_Fall2025.pdf", 2026) == 2025
    assert inputs.guess_year("paper.pdf", 2026) == 2026
    assert inputs.guess_year("old_1999.pdf", 2026) == 2026


def test_predicted_helpers():
    from components import predicted

    assert predicted.hhmmss(10800) == "03:00:00"
    assert predicted.hhmmss(-5) == "00:00:00"
    assert predicted.duration_text(180) == "3 hrs"
    assert predicted.duration_text(60) == "1 hr"
    assert predicted.duration_text(90) == "90 min"
    assert predicted.squares(3) == "\u25a0\u25a0\u25a0\u25a1\u25a1"
    assert predicted.squares(None) == "\u25a1" * 5
    assert predicted.question_label({"number": 2}, 0) == "Q2"
    assert predicted.question_label({"number": "Q4"}, 0) == "Q4"
    text = predicted.realised_text(
        {"realised": {"coding_questions": 1, "problem_solving_questions": 2, "explore_questions": 1}}
    )
    assert text == "1 coding question, 2 problem-solving questions, 1 surprise question."


def test_split_blocks():
    from components import solved

    parts = solved.split_blocks("Use a queue.\n```python\nx = 1\n```\nDone.")
    assert [kind for kind, _ in parts] == ["text", "code", "text"]
    assert parts[1][1] == "x = 1"
    assert solved.split_blocks("plain text") == [("text", "plain text")]


def test_timer_tone():
    from components import timer

    assert timer.tone(3600) == ""
    assert timer.tone(600) == "amber"
    assert timer.tone(300) == "red"
    assert timer.hhmmss(3661) == "01:01:01"


def test_html_is_escaped():
    import theme

    assert "<script>" not in theme.esc("<script>alert(1)</script>")


# ---------- pages ----------
def open_page(rel: str) -> AppTest:
    """Run a page. Uses switch_page when this Streamlit version offers it, else runs the file directly."""
    if rel != "Home.py" and hasattr(AppTest, "switch_page"):
        at = AppTest.from_file(str(APP / "Home.py"), default_timeout=TIMEOUT)
        at.run()
        at.switch_page(rel)
        at.run()
        return at
    at = AppTest.from_file(str(APP / rel), default_timeout=TIMEOUT)
    at.run()
    return at


@needs_bundle
@pytest.mark.parametrize("rel", PAGES)
def test_every_page_runs(rel):
    at = open_page(rel)
    assert not at.exception, [e.value for e in at.exception]


@needs_bundle
def test_past_papers_open_subject_and_back():
    at = open_page("pages/1_Past_Papers.py")
    at.button(key="open_0").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["past_subject"]
    at.button(key="past_back").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["past_subject"] is None


@needs_bundle
def test_upcoming_generate_exam_and_solution():
    at = open_page("pages/2_Upcoming_Paper.py")
    at.button(key="up_generate").click().run()
    assert not at.exception, [e.value for e in at.exception]
    if at.session_state["up_screen"] != "paper":
        pytest.skip("the generator declined this subject/exam (low confidence)")
    at.button(key="up_start_exam").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["up_screen"] == "exam"
    at.button(key="ex_exit").click().run()
    assert at.session_state["up_screen"] == "paper"
    at.button(key="up_see_solution").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["up_screen"] == "solved"