"""Feature 5b: weak-area analysis and the "My results" history.

    analyse(paper, grades) -> dict          what went well / badly, per topic, difficulty and type
    ResultsStore().add(paper, grades, analysis) -> attempt id
    ResultsStore().attempts(subject=None) / .weakness(subject) / .delete(id)

analyse() returns
  {"overall": {"score", "total", "percent", "questions", "counted"},
   "topics":  [{"topic_id", "topic_label", "score", "marks", "percent", "questions", "unanswered",
                "band": "weak"|"ok"|"strong", "weakness" (0..1), "confidence": "low"|"ok"}],  weakest first
   "weak_topic_ids": [..], "by_difficulty": {...}, "by_qtype": {...},
   "not_counted": [question numbers that are not marked or unreadable],
   "summary": [plain sentences for the UI]}

Percentages come from the marks that were actually given; a topic with a single question has
low confidence and the summary says so. Attempts are stored on this computer only
(data/cache/results.json); they are the student's own data and are never uploaded.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.inference import config as C

WEAK_BELOW = 50.0          # percent: under this a topic is "weak"
STRONG_FROM = 75.0         # percent: from this a topic is "strong"
COUNTED = ("graded_ai", "manual", "unanswered")


def _band(percent: float) -> str:
    return "weak" if percent < WEAK_BELOW else "strong" if percent >= STRONG_FROM else "ok"


def _group(rows: list[dict], key: str) -> dict:
    acc: dict = {}
    for g in rows:
        k = g.get(key)
        if k is None:
            continue
        a = acc.setdefault(k, {"score": 0.0, "marks": 0.0, "questions": 0})
        a["score"] += g["score"]
        a["marks"] += g["marks"]
        a["questions"] += 1
    return {k: {**v, "percent": round(100 * v["score"] / v["marks"], 1) if v["marks"] else 0.0}
            for k, v in acc.items()}


def analyse(paper: dict, grades: list[dict]) -> dict:
    counted = [g for g in grades if g["status"] in COUNTED and g["score"] is not None]
    if not counted:
        raise ValueError("nothing is marked yet: mark at least one answer first")
    not_counted = [g["number"] for g in grades if g not in counted]
    total = sum(g["marks"] for g in counted)
    score = sum(g["score"] for g in counted)
    overall = {"score": round(score, 1), "total": round(total, 1),
               "percent": round(100 * score / total, 1) if total else 0.0,
               "questions": len(grades), "counted": len(counted)}

    topics = []
    by_topic: dict = {}
    for g in counted:
        if g.get("topic_id") is not None:
            by_topic.setdefault(int(g["topic_id"]), []).append(g)
    for tid, rows in by_topic.items():
        marks = sum(r["marks"] for r in rows)
        sc = sum(r["score"] for r in rows)
        pct = round(100 * sc / marks, 1) if marks else 0.0
        topics.append({"topic_id": tid, "topic_label": rows[0].get("topic_label") or f"topic {tid}",
                       "score": round(sc, 1), "marks": round(marks, 1), "percent": pct,
                       "questions": len(rows), "unanswered": sum(r["status"] == "unanswered" for r in rows),
                       "band": _band(pct), "weakness": round(1 - pct / 100, 3),
                       "confidence": "low" if len(rows) == 1 else "ok"})
    topics.sort(key=lambda t: (-t["weakness"], -t["marks"], t["topic_id"]))
    weak = [t["topic_id"] for t in topics if t["band"] == "weak"]

    summary = [f"You scored {overall['score']:g} out of {overall['total']:g} ({overall['percent']:.0f}%)."]
    if weak:
        names = "; ".join(t["topic_label"] for t in topics if t["band"] == "weak")
        summary.append(f"Weak areas ({len(weak)}): {names}.")
    else:
        summary.append("No weak area stood out in this paper.")
    strong = [t["topic_label"] for t in topics if t["band"] == "strong"]
    if strong:
        summary.append("Strong areas: " + "; ".join(strong) + ".")
    if any(t["confidence"] == "low" and t["band"] == "weak" for t in topics):
        summary.append("Some weak areas rest on a single question, so treat them as hints, not facts.")
    if not_counted:
        summary.append("Not counted (not marked or unreadable): question(s) "
                       + ", ".join(map(str, not_counted)) + ".")
    return {"overall": overall, "topics": topics, "weak_topic_ids": weak,
            "by_difficulty": _group(counted, "difficulty"), "by_qtype": _group(counted, "qtype"),
            "not_counted": not_counted, "summary": summary}


# ---------------------------------------------------------------------- history
class ResultsStore:
    """Attempts of this student, saved as JSON next to the other caches."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else Path(C.CACHE_DIR) / "results.json"

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _save(self, rows: list[dict]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, paper: dict, grades: list[dict], analysis: dict | None = None) -> str:
        analysis = analysis or analyse(paper, grades)
        rows = self._load()
        aid = uuid.uuid4().hex[:10]
        rows.append({"id": aid, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "subject": paper["subject"], "exam_kind": paper["exam_kind"],
                     "title": paper["title"], "paper_source": paper.get("source"),
                     "personalised": bool(paper.get("personalised")),
                     "overall": analysis["overall"],
                     "topics": [{k: t[k] for k in ("topic_id", "topic_label", "score", "marks", "percent",
                                                    "questions")} for t in analysis["topics"]],
                     "questions": [{k: g.get(k) for k in ("number", "topic_id", "marks", "score", "status",
                                                           "difficulty", "qtype")} for g in grades]})
        self._save(rows)
        return aid

    def attempts(self, subject: str | None = None) -> list[dict]:
        rows = [r for r in self._load() if subject is None or r["subject"] == subject]
        return rows[::-1]                      # the file is in the order they were added: newest first

    def delete(self, attempt_id: str) -> bool:
        rows = self._load()
        keep = [r for r in rows if r["id"] != attempt_id]
        if len(keep) == len(rows):
            return False
        self._save(keep)
        return True

    def weakness(self, subject: str, last_n: int = 5, decay: float = 0.6) -> dict[int, dict]:
        """Weak areas over the last attempts of a subject, newest counting most.
        -> {topic_id: {"topic_label", "weakness" (0..1), "percent", "attempts"}}"""
        acc: dict[int, dict] = {}
        for k, att in enumerate(self.attempts(subject)[:last_n]):
            w = decay ** k
            for t in att["topics"]:
                a = acc.setdefault(int(t["topic_id"]), {"topic_label": t["topic_label"], "s": 0.0,
                                                        "m": 0.0, "attempts": 0})
                a["s"] += w * t["score"]
                a["m"] += w * t["marks"]
                a["attempts"] += 1
        return {tid: {"topic_label": a["topic_label"], "weakness": round(1 - a["s"] / a["m"], 3),
                      "percent": round(100 * a["s"] / a["m"], 1), "attempts": a["attempts"]}
                for tid, a in acc.items() if a["m"] > 0}
