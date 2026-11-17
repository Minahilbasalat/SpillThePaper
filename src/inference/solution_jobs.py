"""Writes the missing solutions of a paper in the background.

    job = SolutionJob(service, paper)      # archive solutions are ready at once; the AI starts
                                           # writing the others in parallel threads right away
    job.snapshot()  -> one entry per question: {number, text, marking_scheme, source, note}
    job.pending     -> how many solutions are still being written
    job.total_ai    -> how many needed the AI

While a solution is being written its entry has source "none" and the note says so, so a page can
show it as it is and replace it when it arrives. A solution the AI could not write keeps the
reason in its note and is retried the next time a job is made for the paper (failures are not
cached). Results are cached on disk by SolutionService, so repeats are instant. Threads never
touch Streamlit.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

WAITING = "This solution is being written by the AI. Wait a moment, it will appear here by itself."
MAX_WORKERS = 2                      # free AI plans limit requests per minute: more threads would not help


class SolutionJob:
    def __init__(self, service, paper: dict, workers: int = MAX_WORKERS):
        self.paper = paper
        self.service = service
        self._lock = threading.Lock()
        self._items = []
        for q in paper["questions"]:
            try:
                r = service.solve(paper["subject"], q, allow_ai=False)
            except Exception as e:
                r = {"text": "", "marking_scheme": None, "source": "none", "note": f"No solution available ({e})."}
            self._items.append({"number": q["number"], **r})
        self._waiting = set()
        if service.ai_available():
            self._waiting = {i for i, it in enumerate(self._items)
                             if it["source"] == "none" and not paper["questions"][i].get("low_quality")}
        self.total_ai = len(self._waiting)
        if self._waiting:
            pool = ThreadPoolExecutor(max_workers=min(workers, len(self._waiting)), thread_name_prefix="solve")
            for i in sorted(self._waiting):
                pool.submit(self._write, i)
            pool.shutdown(wait=False)

    def _write(self, i: int) -> None:
        q = self.paper["questions"][i]
        try:
            r = self.service.solve(self.paper["subject"], q, allow_ai=True)
        except Exception as e:
            r = {"text": "", "marking_scheme": None, "source": "none", "note": f"The AI could not write this one ({e})."}
        if r["source"] == "none":
            r = dict(r, note=(r.get("note") or "The AI could not write this solution.") +
                     " Open this page again in a minute to retry.")
        with self._lock:
            self._items[i] = {"number": q["number"], **r}
            self._waiting.discard(i)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._waiting)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(it, note=WAITING) if i in self._waiting else dict(it)
                    for i, it in enumerate(self._items)]
