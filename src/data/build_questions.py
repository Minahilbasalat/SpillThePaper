"""Segment every readable clean paper into questions.

Writes:
  data/interim/questions.csv                  (question text, NOT in Git)
  data/interim/questions_dropped.csv          (exam rules that were removed, NOT in Git)
  data/processed/question_segmentation_summary.csv  (counts only, in Git)

Usage:
  python -m src.data.build_questions
  python -m src.data.build_questions --sample 5   # print 5 random papers to inspect
"""
import argparse
from pathlib import Path

import pandas as pd

from src.data.clean_questions import is_instruction, normalised, strip_furniture
from src.data.load_text import load_papers_with_text
from src.data.segment_questions import segment_questions

ROOT = Path(__file__).resolve().parents[2]
OUT_QUESTIONS = ROOT / "data" / "interim" / "questions.csv"
OUT_SUMMARY = ROOT / "data" / "processed" / "question_segmentation_summary.csv"
OUT_DROPPED = ROOT / "data" / "interim" / "questions_dropped.csv"      # audit trail, NOT in Git


def build_questions(papers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in papers.itertuples(index=False):
        questions, method = segment_questions(rec.text)
        for i, q in enumerate(questions, start=1):
            rows.append({
                "file_id": rec.file_id,
                "subject": rec.subject,
                "year": rec.year,
                "assessment_type": rec.assessment_type,
                "is_solution": rec.is_solution,
                "q_index": i,
                "method": method,
                "q_text": strip_furniture(q),
            })
    return pd.DataFrame(rows)


def drop_instructions(questions: pd.DataFrame):
    """Remove exam rules that were mistaken for questions. Returns (kept, dropped)."""
    norm = questions["q_text"].map(normalised)
    repeats = questions.groupby([questions["subject"], norm])["file_id"].transform("nunique")
    drop = pd.Series([is_instruction(t, r) for t, r in zip(questions["q_text"], repeats)],
                     index=questions.index)
    kept = questions[~drop].copy()
    kept["q_index"] = kept.groupby("file_id").cumcount() + 1
    return kept.reset_index(drop=True), questions[drop].reset_index(drop=True)


def summarise(papers: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    per_paper = questions.groupby("file_id").size()
    method_per_paper = questions.groupby("file_id")["method"].first().value_counts()
    rows = [
        ("papers_with_text_and_year", len(papers)),
        ("papers_with_at_least_one_question", int(per_paper.shape[0])),
        ("total_questions", len(questions)),
        ("questions_per_paper_mean", round(float(per_paper.mean()), 2)),
        ("questions_per_paper_median", float(per_paper.median())),
        ("questions_per_paper_max", int(per_paper.max())),
    ]
    rows += [(f"papers_segmented_by_{m}", int(c)) for m, c in method_per_paper.items()]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="print this many random papers segmented, then exit")
    args = parser.parse_args()

    papers = load_papers_with_text()
    if args.sample:
        for rec in papers.sample(args.sample, random_state=1).itertuples(index=False):
            qs, method = segment_questions(rec.text)
            print("=" * 70)
            print(rec.subject, rec.year, rec.assessment_type, "| method:", method,
                  "| questions:", len(qs))
            for q in qs[:6]:
                print("  -", q[:130])
        return

    questions, dropped = drop_instructions(build_questions(papers))
    OUT_QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
    questions.to_csv(OUT_QUESTIONS, index=False)
    dropped.to_csv(OUT_DROPPED, index=False)
    summary = summarise(papers, questions)
    summary.loc[len(summary)] = ("instructions_dropped", len(dropped))
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))
    print("saved:", OUT_QUESTIONS)


if __name__ == "__main__":
    main()
