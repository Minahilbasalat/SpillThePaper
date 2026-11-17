"""Solutions written in the background while the student reads the paper."""
import threading
import time

from src.inference.solution_jobs import WAITING, SolutionJob
from src.inference.solutions import SolutionService

PAPER = {"subject": "CN", "questions": [
    {"number": 1, "text": "Explain TCP congestion control in detail.", "marks": 5, "solution": "From the paper.",
     "solution_source": "inline", "low_quality": False},
    {"number": 2, "text": "Explain how UDP differs from TCP in detail.", "marks": 5, "solution": None, "low_quality": False},
    {"number": 3, "text": "Describe the three way handshake step by step.", "marks": 5, "solution": None, "low_quality": False},
    {"number": 4, "text": "x ?? ??? garbled scan text here", "marks": 5, "solution": None, "low_quality": True}]}


class Slow:
    available, model = True, "m"

    def __init__(self, delay=0.15, fail_on=None):
        self.delay, self.fail_on, self.threads, self.gate = delay, fail_on, set(), threading.Event()

    def complete(self, system, user, max_tokens=1200):
        self.threads.add(threading.current_thread().name)
        self.gate.wait(2)
        time.sleep(self.delay)
        if self.fail_on and self.fail_on in user:
            raise RuntimeError("quota")
        return '{"solution": "Answer for: ' + user.split("Question:")[1].strip()[:20] + '", "marking_scheme": null}'


class Off:
    available, model = False, "none"


def wait(job, seconds=10):
    end = time.time() + seconds
    while job.pending and time.time() < end:
        time.sleep(0.02)


def test_archive_is_instant_and_the_ai_fills_the_rest_in_parallel(tmp_path):
    ai = Slow()
    job = SolutionJob(SolutionService(ai, tmp_path), PAPER)
    first = job.snapshot()
    assert first[0]["source"] == "archive" and first[0]["number"] == 1
    assert first[1]["source"] == "none" and first[1]["note"] == WAITING and job.pending == 2 and job.total_ai == 2
    assert "garbled" in first[3]["note"] and first[3]["note"] != WAITING                 # bad scans are not attempted
    ai.gate.set()
    wait(job)
    done = job.snapshot()
    assert job.pending == 0 and done[1]["source"] == "ai" and done[2]["source"] == "ai"
    assert done[1]["text"].startswith("Answer for: Explain how UDP") and len(ai.threads) == 2   # two threads at once


def test_second_job_is_served_from_the_disk_cache(tmp_path):
    ai = Slow(delay=0.01)
    ai.gate.set()
    wait(SolutionJob(SolutionService(ai, tmp_path), PAPER))
    ai2 = Slow()
    ai2.gate.set()
    job = SolutionJob(SolutionService(ai2, tmp_path), PAPER)
    wait(job)
    assert not ai2.threads and all(e["source"] in ("archive", "ai", "none") for e in job.snapshot())
    assert job.snapshot()[1]["source"] == "ai"


def test_failures_are_reported_not_cached_and_do_not_stop_the_others(tmp_path):
    ai = Slow(fail_on="UDP")
    ai.gate.set()
    job = SolutionJob(SolutionService(ai, tmp_path), PAPER)
    wait(job)
    s = job.snapshot()
    assert s[1]["source"] == "none" and "could not write" in s[1]["note"] and "again" in s[1]["note"] and s[2]["source"] == "ai"
    again = Slow(fail_on="nothing")
    again.gate.set()
    job2 = SolutionJob(SolutionService(again, tmp_path), PAPER)
    wait(job2)
    assert job2.snapshot()[1]["source"] == "ai"                                          # retried next time


def test_without_a_key_nothing_is_waiting_and_the_note_says_so(tmp_path):
    job = SolutionJob(SolutionService(Off(), tmp_path), PAPER)
    s = job.snapshot()
    assert job.pending == 0 and job.total_ai == 0 and s[1]["source"] == "none"
    assert "switched off (no API key is set)" in s[1]["note"] and s[0]["source"] == "archive"
