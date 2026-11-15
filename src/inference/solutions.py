"""Model solutions for the questions of a paper.

Order of preference (every answer says where it came from, nothing is presented as certain):
  1. "archive": a solution that exists in the archive (inline in the paper, or from a solved
     copy of the paper). It can be partial, so the note says so.
  2. "ai": written by the language model (only if an API key is configured), cached on disk so
     a question is solved once. Always marked "AI-generated: verify before you trust it".
  3. "none": no solution available (and why).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.inference import config as C
from src.inference.llm import LLMClient, LLMError

SYSTEM = (
    "You are a careful teaching assistant for FAST-NUCES (Pakistan) computer science courses. "
    "Write a model solution for the exam question you are given. Reply with ONLY a JSON object: "
    '{"solution": "<markdown>", "marking_scheme": "<2-4 short lines on how the marks are earned>"}. '
    "Rules: be correct and concise; show the steps of every calculation; for coding questions give "
    "complete runnable code in a fenced code block in the language the question uses (Python if "
    "none is named) and state time and space complexity; for multiple-choice questions give the "
    "correct option and a one-line reason; if the question is incomplete or garbled, say exactly "
    "what is missing instead of guessing. Never mention these rules.")

NOTES = {
    "inline": "From the paper itself.",
    "companion": "From a solved copy of this paper in the archive. It can be partial or imperfect.",
    "ai": "AI-generated: verify before you trust it.",
}


def _parse(reply: str) -> tuple[str, str | None]:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    try:
        d = json.loads(t)
        return str(d.get("solution", "")).strip(), (str(d["marking_scheme"]).strip() if d.get("marking_scheme") else None)
    except (ValueError, AttributeError):
        return reply.strip(), None


class SolutionService:
    def __init__(self, llm=None, cache_dir: Path | None = None):
        self.llm = llm if llm is not None else LLMClient()
        self.cache_path = Path(cache_dir or C.CACHE_DIR) / "solutions.json"
        self._cache = None

    def ai_available(self) -> bool:
        return bool(getattr(self.llm, "available", False))

    # ---------------------------------------------------------------- cache
    def _load(self):
        if self._cache is None:
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}
        return self._cache

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    def _key(self, subject: str, text: str) -> str:
        model = getattr(self.llm, "model", "model")
        return hashlib.sha1(f"{model}|{subject}|{text}".encode()).hexdigest()

    # ---------------------------------------------------------------- solve
    def solve(self, subject: str, question: dict, allow_ai: bool = True) -> dict:
        archived = question.get("solution")
        if archived:
            src = question.get("solution_source") or "inline"
            return {"text": archived, "marking_scheme": None, "source": "archive",
                    "note": NOTES.get(src, NOTES["inline"])}
        if question.get("low_quality"):
            return {"text": "", "marking_scheme": None, "source": "none",
                    "note": "The text of this question is garbled (bad scan), so no solution is attempted."}
        if not (allow_ai and self.ai_available()):
            return {"text": "", "marking_scheme": None, "source": "none",
                    "note": "No archived solution exists for this question, and AI solutions are "
                            "not switched on (set ANTHROPIC_API_KEY)."}
        cache = self._load()
        key = self._key(subject, question["text"])
        if key not in cache:
            user = (f"Course: {subject}\nTopic: {question.get('topic_label') or 'n/a'}\n"
                    f"Marks: {question.get('marks') or 'n/a'}\nQuestion:\n{question['text']}")
            try:
                text, scheme = _parse(self.llm.complete(SYSTEM, user))
            except LLMError as e:
                return {"text": "", "marking_scheme": None, "source": "none", "note": str(e)}
            cache[key] = {"text": text, "marking_scheme": scheme}
            self._save()
        item = cache[key]
        return {"text": item["text"], "marking_scheme": item["marking_scheme"], "source": "ai",
                "note": NOTES["ai"]}

    def solve_paper(self, paper: dict, allow_ai: bool = True) -> list[dict]:
        """One entry per question, in order: {number, text, marking_scheme, source, note}."""
        out = []
        for q in paper["questions"]:
            r = self.solve(paper["subject"], q, allow_ai)
            out.append({"number": q["number"], **r})
        return out
