"""The solved-paper page: solutions appear by themselves while the AI writes them in the background."""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
from src.inference import config as C  # noqa: E402
from src.inference.solution_jobs import WAITING, SolutionJob  # noqa: E402
from src.inference.solutions import SolutionService  # noqa: E402

pytestmark = pytest.mark.skipif(not C.QUESTION_TABLE_CSV.exists(), reason="runtime bundle missing")

PAPER = {"title": "Predicted Mid 1 2026 - CN", "subject": "CN", "exam_kind": "mid1", "year": 2026, "source": "predicted",
         "assessment_type": "mid", "duration_minutes": 60, "total_marks": 20, "notes": ["note"], "topics": None,
         "realised": {}, "blueprint": {}, "questions": [
             {"number": 1, "text": "Explain TCP congestion control in detail.", "marks": 10, "qtype": "theory",
              "difficulty": "easy", "topic_id": 1, "topic_label": "tcp", "solution": "From the paper itself.",
              "solution_source": "inline", "low_quality": False, "explore": False, "reason": "r", "likelihood": "Likely",
              "source": "Mid 1 2023"},
             {"number": 2, "text": "Explain how UDP differs from TCP in detail.", "marks": 10, "qtype": "theory",
              "difficulty": "easy", "topic_id": 2, "topic_label": "udp", "solution": None, "solution_source": None,
              "low_quality": False, "explore": False, "reason": "r", "likelihood": "Likely", "source": "Mid 1 2022"}]}


class Slow:
    available, model = True, "m"

    def __init__(self):
        self.gate = threading.Event()

    def complete(self, system, user, max_tokens=1200):
        self.gate.wait(5)
        return '{"solution": "UDP has no handshake and no retransmission.", "marking_scheme": null}'


class Off:
    available, model = False, "none"


def open_solved(tmp_path, llm):
    sys.path[:0] = [str(ROOT), str(ROOT / "app")]
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180)
    at.run()
    job = SolutionJob(SolutionService(llm, tmp_path), PAPER)
    for k, v in {"up_paper": PAPER, "up_screen": "solved", "up_revealed": [], "up_pdf": None, "up_exam_timer": False,
                 "up_solutions": {"job": job, "paper": PAPER, "pdf": None}}.items():
        at.session_state[k] = v
    at.switch_page("pages/2_Upcoming_Paper.py")
    at.run()
    return at, job


def page_text(at):
    return " ".join(str(x.value) for x in list(at.markdown) + list(at.caption))


def test_waiting_text_then_the_solution_appears(tmp_path):
    ai = Slow()
    at, job = open_solved(tmp_path, ai)
    assert not at.exception
    text = page_text(at)
    assert "From the paper itself." in text and WAITING[:40] in text
    assert not [b for b in at.button if "Generate missing" in str(b.label)]                  # the button is gone
    assert "up_solved_pdf_btn" not in [getattr(b, "key", None) for b in at.get("download_button")]
    ai.gate.set()
    end = time.time() + 10
    while job.pending and time.time() < end:
        time.sleep(0.02)
    at.run()
    assert not at.exception and "UDP has no handshake" in page_text(at)


def test_no_key_says_so(tmp_path):
    at, job = open_solved(tmp_path, Off())
    assert not at.exception and job.pending == 0
    assert "switched off (no API key is set)" in page_text(at)
