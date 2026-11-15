"""Attach archived solutions to questions.

The archive often holds a paper AND a solved copy of it. Step 6 of the data pipeline set the
solved copy aside ("solution_companion") and recorded which paper it solves (duplicate_of).
Here every question of that paper is matched to the most similar chunk of the solved copy;
the chunk minus the question statement is the solution.

    python -m src.data.pair_solutions        -> data/interim/solutions_paired.csv

A pair is accepted only when the question and the chunk are clearly the same question
(cosine >= MIN_SIM) and some answer text remains after removing the statement.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.clean_questions import strip_furniture
from src.data.load_text import build_text_index, find_text_path, read_text
from src.data.segment_questions import segment_questions

ROOT = Path(__file__).resolve().parents[2]
FLAGGED = ROOT / "data" / "processed" / "papers_flagged.csv"
QUESTIONS = ROOT / "data" / "interim" / "questions.csv"
OUT = ROOT / "data" / "interim" / "solutions_paired.csv"
MIN_SIM = 0.55
MIN_SOLUTION_WORDS = 6


def remove_statement(chunk: str, question: str) -> str:
    """The chunk of a solved paper = the question statement followed by the answer.
    Drop the longest run of words shared with the question that sits at the start."""
    cw, qw = chunk.split(), question.split()
    sm = difflib.SequenceMatcher(None, cw[:len(qw) + 40], qw, autojunk=False)
    end = 0
    for a, _b, size in sm.get_matching_blocks():
        if size >= 4 and a <= end + 6:            # contiguous with what we already removed
            end = a + size
    return " ".join(cw[end:]).strip()


def pair_all() -> pd.DataFrame:
    flagged = pd.read_csv(FLAGGED, dtype={"file_id": str, "duplicate_of": str})
    comp = flagged[flagged["exclusion_reason"] == "solution_companion"]
    q = pd.read_csv(QUESTIONS, dtype={"file_id": str}).dropna(subset=["q_text"])
    by_paper = {f: g for f, g in q.groupby("file_id")}
    index = build_text_index()
    rows = []
    for c in comp.itertuples(index=False):
        paper_q = by_paper.get(c.duplicate_of)
        path = find_text_path(c.file_id, index) if c.has_text else None
        if paper_q is None or path is None:
            continue
        chunks = [strip_furniture(x) for x in segment_questions(read_text(path))[0]]
        chunks = [x for x in chunks if len(x.split()) >= 6]
        if not chunks:
            continue
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(
            list(paper_q["q_text"]) + chunks)
        sim = (vec.transform(paper_q["q_text"]) @ vec.transform(chunks).T).toarray()
        for i, r in enumerate(paper_q.itertuples(index=False)):
            j = int(sim[i].argmax())
            if sim[i, j] < MIN_SIM:
                continue
            sol = remove_statement(chunks[j], r.q_text)
            if len(sol.split()) >= MIN_SOLUTION_WORDS:
                rows.append({"file_id": c.duplicate_of, "q_index": int(r.q_index),
                             "solution_text": sol, "solution_file_id": c.file_id,
                             "similarity": round(float(sim[i, j]), 3)})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("similarity", ascending=False).drop_duplicates(["file_id", "q_index"])
    return out


def main():
    out = pair_all()
    out.to_csv(OUT, index=False, encoding="utf-8")
    q = pd.read_csv(QUESTIONS, dtype={"file_id": str})
    print(f"solutions paired to {len(out)} questions of {len(q)} "
          f"({len(out) / max(len(q), 1):.1%}) -> {OUT}")


if __name__ == "__main__":
    main()
