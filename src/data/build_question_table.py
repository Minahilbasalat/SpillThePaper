"""Build data/interim/question_table.csv (Stage 2) and a per-paper blueprint table.

    python -m src.data.build_question_table
"""
import pandas as pd

from src.data.question_features import build_question_table

SRC = "data/interim/questions_topics.csv"
OUT = "data/interim/question_table.csv"
BLUEPRINT = "data/processed/paper_blueprints.csv"


def main():
    q = pd.read_csv(SRC, dtype={"file_id": str})
    t = build_question_table(q)
    t.to_csv(OUT, index=False, encoding="utf-8")

    n = len(t)
    print(f"questions: {n}")
    print(f"marks parsed: {t['marks'].notna().mean():.1%}  ({t['marks_source'].value_counts().to_dict()})")
    print(f"answer separated: {t['has_solution'].mean():.1%}; "
          f"possibly contaminated (solution paper, no split): {t['contaminated'].mean():.1%}")
    print("question types:", t["qtype"].value_counts(normalize=True).round(3).to_dict())
    print("Bloom levels:", t["bloom"].value_counts(normalize=True).sort_index().round(3).to_dict())
    print("difficulty by type:\n", t.groupby("assessment_type")["difficulty_score"].mean().round(2))

    # 100 questions (stratified by predicted type) for a MANUAL accuracy check of the rules
    usable_q = t[~t["contaminated"]]
    parts = [g.sample(min(len(g), 12), random_state=1) for _, g in usable_q.groupby("qtype")]
    sample = pd.concat(parts).sample(frac=1, random_state=1).head(100)
    sample[["file_id", "q_index", "subject", "assessment_type", "question_text", "qtype",
            "marks", "marks_source", "bloom"]].assign(true_qtype="", true_marks="").to_csv(
        "data/processed/question_label_sample.csv", index=False, encoding="utf-8")

    # one row per paper: what a paper of this kind looks like (used by the paper generator)
    usable = t[~t["contaminated"]]
    g = usable.groupby("file_id")
    bp = g.agg(subject=("subject", "first"), year=("year", "first"),
               assessment_type=("assessment_type", "first"), n_questions=("q_index", "count"),
               marks_known=("marks", lambda s: s.notna().mean()),
               marks_sum=("marks", "sum"), mean_bloom=("bloom", "mean")).reset_index()
    for qt in sorted(t["qtype"].unique()):
        bp[f"share_{qt}"] = g["qtype"].apply(lambda s, qt=qt: (s == qt).mean()).values
    bp.to_csv(BLUEPRINT, index=False, encoding="utf-8")
    print(f"paper blueprints: {len(bp)} -> {BLUEPRINT}")


if __name__ == "__main__":
    main()