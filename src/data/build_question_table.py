"""Build data/interim/question_table.csv (Stage 2) and a per-paper blueprint table.

    python -m src.data.build_question_table
"""
import pandas as pd

from src.data.exam_kinds import exam_kind
from src.data.question_features import build_question_table
from src.data.text_quality import build_known_words, save_known_words

SRC_ALL = "data/interim/questions.csv"                 # every subject
SRC_TOPICS = "data/interim/questions_topics.csv"       # topic ids (supported subjects only)
PAPERS = "data/processed/papers_clean.csv"
PAIRED = "data/interim/solutions_paired.csv"          # from src.data.pair_solutions (optional)
OUT = "data/interim/question_table.csv"
BLUEPRINT = "data/processed/paper_blueprints.csv"


def main():
    q = pd.read_csv(SRC_ALL, dtype={"file_id": str}).dropna(subset=["q_text"])
    topics = pd.read_csv(SRC_TOPICS, dtype={"file_id": str})[["file_id", "q_index", "topic_id"]]
    q = q.merge(topics, on=["file_id", "q_index"], how="left")       # topic_id empty = unsupported subject
    papers = pd.read_csv(PAPERS, dtype={"file_id": str})[["file_id", "assessment_number", "term"]]
    q = q.merge(papers, on="file_id", how="left")
    q["exam_kind"] = [exam_kind(a, n) for a, n in zip(q["assessment_type"], q["assessment_number"])]
    known = build_known_words(q["q_text"])
    save_known_words(known)
    t = build_question_table(q, known)
    try:                                       # archived solutions found in solved copies of papers
        paired = pd.read_csv(PAIRED, dtype={"file_id": str})[["file_id", "q_index", "solution_text"]]
        paired = paired.rename(columns={"solution_text": "paired_solution"})
        t = t.merge(paired, on=["file_id", "q_index"], how="left")
        use = (~t["has_solution"]) & t["paired_solution"].notna()
        t.loc[use, "solution_text"] = t.loc[use, "paired_solution"]
        t.loc[use, "solution_source"] = "companion"
        t["has_solution"] = t["solution_text"].fillna("").str.len() > 0
        t = t.drop(columns="paired_solution")
        print(f"archived solutions paired to {int(use.sum())} questions")
    except FileNotFoundError:
        print("no solutions_paired.csv yet (run python -m src.data.pair_solutions)")
    t.to_csv(OUT, index=False, encoding="utf-8")

    n = len(t)
    print(f"questions: {n}")
    print(f"marks parsed: {t['marks'].notna().mean():.1%}  ({t['marks_source'].value_counts().to_dict()})")
    print(f"low-quality (garbled) text: {t['low_quality'].mean():.1%}")
    print(f"answer available: {t['has_solution'].mean():.1%}; "
          f"possibly contaminated (solution paper, no split): {t['contaminated'].mean():.1%}")
    print("question types:", t["qtype"].value_counts(normalize=True).round(3).to_dict())
    print("Bloom levels:", t["bloom"].value_counts(normalize=True).sort_index().round(3).to_dict())
    print("difficulty by type:\n", t.groupby("assessment_type")["difficulty_score"].mean().round(2))

    # 100 questions (stratified by predicted type) for a MANUAL accuracy check of the rules
    usable_q = t[(~t["contaminated"]) & (~t["low_quality"])]
    parts = [g.sample(min(len(g), 12), random_state=1) for _, g in usable_q.groupby("qtype")]
    sample = pd.concat(parts).sample(frac=1, random_state=1).head(100)
    sample[["file_id", "q_index", "subject", "assessment_type", "question_text", "qtype",
            "marks", "marks_source", "bloom"]].assign(true_qtype="", true_marks="").to_csv(
        "data/processed/question_label_sample.csv", index=False, encoding="utf-8")

    # one row per paper: what a paper of this kind looks like (used by the paper generator)
    usable = t[(~t["contaminated"]) & (~t["low_quality"])]
    g = usable.groupby("file_id")
    bp = g.agg(subject=("subject", "first"), year=("year", "first"),
               assessment_type=("assessment_type", "first"), exam_kind=("exam_kind", "first"),
               n_questions=("q_index", "count"),
               marks_known=("marks", lambda s: s.notna().mean()),
               marks_sum=("marks", "sum"), mean_bloom=("bloom", "mean")).reset_index()
    for d in ("easy", "medium", "hard"):
        bp[f"diff_{d}"] = g["difficulty"].apply(lambda s, d=d: (s == d).mean()).values
    for qt in sorted(t["qtype"].unique()):
        bp[f"share_{qt}"] = g["qtype"].apply(lambda s, qt=qt: (s == qt).mean()).values
    bp.to_csv(BLUEPRINT, index=False, encoding="utf-8")
    print(f"paper blueprints: {len(bp)} -> {BLUEPRINT}")


if __name__ == "__main__":
    main()
