"""Build a tentative upcoming paper from REAL data.

    gen = PaperGenerator(predictor, library)
    paper = gen.generate("Computer Networks", "mid1", 2026,
                         syllabus_items=[...], extra=UploadResult, strict_syllabus=False)

How the paper is made (all of it is also written into paper["notes"]):
  1. Structure: number of questions, total marks, question-type mix and difficulty mix come
     from the real past papers of this subject and exam kind (Library.blueprint).
  2. Topics: Predictor.predict gives a probability per topic. Most questions go to the most
     likely topics; about 20% of the questions go to LESS-PREDICTED topics so that surprises
     are not missed.
  3. Questions: for every slot a real past question of that topic is chosen to match the
     wanted type and difficulty, preferring recent ones and (if a syllabus is given) ones that
     match the syllabus. These are past questions, not newly written ones.
  4. Marks are scaled so that the paper adds up to the typical total marks of that exam.
The result is reproducible: the same inputs and seed give the same paper.
"""
from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd

from src.data.exam_kinds import label
from src.inference.library import default_duration
from src.inference.syllabus import SyllabusMatcher

DEFAULT_TOTAL = {"quiz": 15.0, "mid": 30.0, "final": 70.0}
DIFF_RANK = {"easy": 0, "medium": 1, "hard": 2}
POOL_COLS = ["file_id", "q_index", "year", "exam_kind", "question_text", "marks", "qtype",
             "difficulty", "bloom", "topic_id", "solution_text", "solution_source"]
_LEET = re.compile(r"\b(return|given|input|output|array|list|string|tree|graph|complexity|O\()", re.I)
_TOKEN = re.compile(r"[a-z0-9]+")
_LABEL = re.compile(r"^\s*(?:Q(?:uestion)?\s*\d+\s*[a-z]?\s*[:.)\-–]+\s*|\d+\s*[.)]\s+)", re.I)
MIN_WORDS = 10


def clean_label(text: str) -> str:
    """Remove the original question number ("Q4)", "Question 2:") - questions are renumbered."""
    text = re.sub(r"\[\s*CLO[^\]]*\]\s*", "", text, flags=re.I)         # course-outcome tags
    return _LABEL.sub("", text, count=1).strip()

def fit_marks(raw, total) -> tuple[np.ndarray, int]:
    """Whole-number marks, every question >= 1, adding up to EXACTLY the returned total."""
    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    total = max(int(round(total)), n)
    base = raw * total / raw.sum() if raw.sum() > 0 else np.full(n, total / n)
    m = np.maximum(1, np.round(base)).astype(int)
    while m.sum() > total:
        i = int(np.argmax(m))
        if m[i] <= 1:
            break
        m[i] -= 1
    while m.sum() < total:
        i = int(np.argmax(base - m))
        m[i] += 1
    return m.astype(float), total

def _jaccard(a: str, b: str) -> float:
    x, y = set(_TOKEN.findall(a.lower())), set(_TOKEN.findall(b.lower()))
    return len(x & y) / len(x | y) if x and y else 0.0


class PaperGenerator:
    def __init__(self, predictor, library):
        self.pred, self.lib = predictor, library

    # ------------------------------------------------------------------ pool
    def _pool(self, subject, extra):
        u = self.lib.usable
        pool = u[(u["subject"] == subject) & u["topic_id"].notna()][POOL_COLS].copy()
        if extra is not None and len(extra.questions):
            eq = extra.questions
            eq = eq[(~eq["contaminated"].astype(bool)) & (eq["qtype"] != "fragment")
                    & (~eq["low_quality"].astype(bool))]
            pool = pd.concat([pool, eq[POOL_COLS]], ignore_index=True)
        pool["topic_id"] = pool["topic_id"].astype(int)
        pool["question_text"] = pool["question_text"].map(clean_label)
        pool = pool[pool["question_text"].str.split().str.len() >= MIN_WORDS]
        return pool.reset_index(drop=True)

    # -------------------------------------------------------------- generate
    def generate(self, subject: str, exam: str, year: int, model: str = "logreg",
                 syllabus_items=None, extra=None, n_questions: int | None = None,
                 total_marks: float | None = None, explore_share: float = 0.2,
                 strict_syllabus: bool = False, seed: int | None = None) -> dict:
        exam = exam.lower()
        atype = "mid" if exam.startswith("mid") else exam
        rng = np.random.default_rng(
            seed if seed is not None else zlib.crc32(f"{subject}|{exam}|{year}|{model}".encode()))
        bp = self.lib.blueprint(subject, exam)
        n = int(n_questions or max(2, round(bp["questions"])))
        total = float(total_marks or bp["total_marks"] or DEFAULT_TOTAL.get(atype, 30.0))
        pred = self.pred.predict(subject, exam, year, model,
                                 None if extra is None else extra.papers,
                                 None if extra is None else extra.incidence)
        pool = self._pool(subject, extra)
        notes = []

        # ---- syllabus: a soft preference (and an optional filter), never a guarantee
        pool["syl"] = np.nan
        topic_w = {}
        items = [s for s in (syllabus_items or []) if s.strip()]
        if items:
            m = SyllabusMatcher(pool["question_text"].tolist(), items)
            pool["syl"] = pd.Series(m.scores(pool["question_text"].tolist())).rank(pct=True).to_numpy()
            if strict_syllabus:
                keep = pool[pool["syl"] >= 0.5]
                if len(keep) >= 2 * n:
                    pool = keep.reset_index(drop=True)
                else:
                    notes.append("Strict syllabus mode left too few questions, so it was relaxed.")
            topic_w = pool.groupby("topic_id")["syl"].mean().apply(lambda v: 0.5 + v).to_dict()
            notes.append("Syllabus matching compares words, so it is approximate: check the "
                         "'syllabus match' of each question.")

        probs = pred.set_index("topic_id")["probability"].to_dict()
        thr = pred.attrs["threshold"]
        has_q = set(pool["topic_id"])
        weight = {t: probs[t] * topic_w.get(t, 1.0) for t in probs if t in has_q}
        if not weight:
            raise ValueError("no questions available for this subject")
        order = sorted(weight, key=weight.get, reverse=True)
        rank_of = {t: i + 1 for i, t in enumerate(order)}

        # ---- slots: main topics (by probability) + exploration topics (less predicted)
        n_explore = 0 if n < 4 else min(n // 2, max(1, round(explore_share * n)))
        n_main = n - n_explore
        topic_slots = []
        remaining = list(weight)
        while len(topic_slots) < n_main:
            if not remaining:
                remaining = list(weight)
            w = np.array([weight[t] ** 2 + 1e-6 for t in remaining])
            t = int(rng.choice(remaining, p=w / w.sum()))
            remaining.remove(t)
            topic_slots.append((t, False))
        low = [t for t in weight if t not in {s[0] for s in topic_slots} and weight[t] < thr]
        low = low or [t for t in weight if t not in {s[0] for s in topic_slots}] or list(weight)
        for _ in range(n_explore):
            t = int(rng.choice(low))
            topic_slots.append((t, True))
            if len(low) > 1:
                low.remove(t)

        # ---- wanted question types and difficulties from the real blueprint
        types = {k: v for k, v in bp["question_type_shares"].items() if k != "fragment" and v > 0}
        tk, tp = list(types), np.array(list(types.values()))
        dk, dp = list(bp["difficulty_mix"]), np.array(list(bp["difficulty_mix"].values()))
        ymin, ymax = pool["year"].min(), pool["year"].max()
        pool["recency"] = (pool["year"] - ymin) / max(ymax - ymin, 1)

        chosen = []
        for topic, explore in topic_slots:
            want_t = str(rng.choice(tk, p=tp / tp.sum()))
            want_d = str(rng.choice(dk, p=dp / dp.sum() if dp.sum() else None))
            cand = pool[pool["topic_id"] == topic]
            cand = cand[~cand["question_text"].isin([c["question_text"] for c in chosen])]
            cand = cand[[all(_jaccard(t, c["question_text"]) < 0.7 for c in chosen)
                         for t in cand["question_text"]]] if len(chosen) else cand
            if cand.empty:                                    # topic exhausted: use the best other topic
                cand = pool[~pool["question_text"].isin([c["question_text"] for c in chosen])]
                cand = cand[[all(_jaccard(t, c["question_text"]) < 0.7 for c in chosen)
                             for t in cand["question_text"]]] if len(chosen) else cand
                if cand.empty:
                    break
            score = (3.0 * (cand["qtype"] == want_t) + 2.0 * (cand["difficulty"] == want_d)
                     - 1.0 * (cand["qtype"] == "other")
                     + 1.5 * cand["syl"].fillna(0) + 0.4 * cand["recency"]
                     + 0.2 * rng.random(len(cand)))
            r = cand.loc[score.idxmax()]
            t_id = int(r["topic_id"])
            chosen.append({
                "question_text": r["question_text"], "marks_src": r["marks"], "qtype": r["qtype"],
                "difficulty": r["difficulty"], "topic_id": t_id, "explore": explore,
                "topic_probability": round(float(probs.get(t_id, 0.0)), 3),
                "topic_rank": rank_of.get(t_id),
                "syllabus_match": None if pd.isna(r["syl"]) else round(float(r["syl"]), 2),
                "source": f"{label(r['exam_kind'])} {int(r['year'])}",
                "solution": r["solution_text"] if isinstance(r["solution_text"], str) and r["solution_text"] else None,
                "solution_source": r["solution_source"] if isinstance(r["solution_text"], str) and r["solution_text"] else None,
            })

        # ---- marks scaled to the typical total of this exam
        known = [c["marks_src"] for c in chosen if pd.notna(c["marks_src"])]
        fill = float(np.median(known)) if known else total / max(len(chosen), 1)
        raw = np.array([c["marks_src"] if pd.notna(c["marks_src"]) else fill for c in chosen], dtype=float)
        raw = np.clip(raw, 0.5 * fill, 2.0 * fill)          # parsed marks are noisy: cap the extremes
        wanted_total = total
        scaled, total = fit_marks(raw, total)
        if total != round(wanted_total):
            notes.append(f"The usual total ({wanted_total:.0f}) is smaller than the number of "
                         f"questions, so every question carries 1 mark (total {total}).")
        for c, m, was in zip(chosen, scaled, [pd.notna(c["marks_src"]) for c in chosen]):
            c["marks"] = float(m)
            c["marks_note"] = "scaled from the original question" if was else "estimated"

        chosen.sort(key=lambda c: (DIFF_RANK[c["difficulty"]], -c["topic_probability"]))
        questions = []
        for i, c in enumerate(chosen, start=1):
            lab = self.lib.topic_label(subject, c["topic_id"])
            reason = ("Less-predicted area, included so surprises are not missed" if c["explore"]
                      else f"Likely topic (rank {c['topic_rank']} of {len(weight)}, "
                           f"probability {c['topic_probability']:.2f})")
            level = max(1, 5 - int(5 * (c["topic_rank"] - 1) / max(len(weight), 1)))
            questions.append({
                "number": i, "text": c["question_text"], "marks": c["marks"],
                "likelihood": 1 if c["explore"] else level,
                "leetcode_style": c["qtype"] == "code" and bool(_LEET.search(c["question_text"])),
                "solution_source": c["solution_source"],
                "marks_note": c["marks_note"], "qtype": c["qtype"], "difficulty": c["difficulty"],
                "topic_id": c["topic_id"], "topic_label": lab, "solution": c["solution"],
                "reason": reason, "explore": c["explore"], "topic_probability": c["topic_probability"],
                "syllabus_match": c["syllabus_match"], "source": c["source"]})

        notes.insert(0, "These are real past questions chosen for the predicted topics; they are "
                        "not newly written questions.")
        notes.append(f"Structure follows {bp['papers_used']} real past {label(exam)} paper(s) "
                     f"({bp['scope']}): about {bp['questions']:.0f} questions, {bp['total_marks'] or total:.0f} marks.")
        if exam in ("mid1", "mid2") and not pred.attrs["phase_adjusted"]:
            notes.append(f"Too few earlier {label(exam)} papers to weight topics by {label(exam)}; "
                         "the general mid prediction is used.")
        if pred.attrs["uploaded_papers"]:
            notes.append(f"{pred.attrs['uploaded_papers']} uploaded paper(s) were added to the history.")
        if pred["examined_this_year"].any():
            notes.append("Topics already examined earlier this year are marked in the topic table.")
        notes.append("Probabilities are rankings, not exact chances (they are over-estimated).")

        cnt = pd.Series([q["difficulty"] for q in questions]).value_counts(normalize=True)
        return {
            "title": f"Predicted {label(exam)} {year} - {subject}", "subject": subject,
            "assessment_type": atype, "exam_kind": exam, "year": int(year), "source": "predicted",
            "duration_minutes": default_duration(exam), "total_marks": float(total),
            "questions": questions, "notes": notes, "blueprint": bp, "topics": pred,
            "realised": {"difficulty_mix": {k: round(float(cnt.get(k, 0)), 2) for k in DIFF_RANK},
                         "coding_questions": sum(q["qtype"] == "code" for q in questions),
                         "problem_solving_questions": sum(q["qtype"] in ("code", "numerical", "proof_derivation") for q in questions),
                         "explore_questions": sum(q["explore"] for q in questions)},
        }
