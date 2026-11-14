"""One-shot diagnostics. Run:  python scripts/diagnose.py   and paste the whole output."""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
I = ROOT / "data" / "interim"
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 110)


def header(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


header("1. files present")
for f in [P / "papers_clean.csv", P / "subject_eligibility.csv", P / "ocr_results.csv",
          I / "questions.csv", I / "questions_topics.csv", I / "question_table.csv"]:
    print(("OK      " if f.exists() else "MISSING ") + str(f.relative_to(ROOT)))

header("2. eligibility summary")
el = pd.read_csv(P / "subject_eligibility.csv")
print("eligible subjects:", int(el["eligible"].sum()), "of", len(el))
print(el.loc[~el["eligible"], "reason"].value_counts().to_string())

header("3. possible duplicate subject names")
papers = pd.read_csv(P / "papers_clean.csv", dtype={"file_id": str})
pairs = [("PDC", "Parallel and Distributed Computing"),
         ("SPM", "Software Process Management and Metrics"),
         ("DW", "Data Warehousing and Data Mining"),
         ("Intro to Data Science", "Data Science"),
         ("SQA", "Software Quality Engineering")]
for a, b in pairs:
    for s in (a, b):
        g = papers[papers["subject"] == s]
        yrs = sorted(g["year"].dropna().astype(int).unique().tolist())
        print(f"\n[{s}] papers={len(g)} years={yrs}")
        print(g["repo_path"].head(3).to_string(index=False))

header("4. Computer Networks Lab: are the questions near-copies of each other?")
q = pd.read_csv(I / "questions.csv", dtype={"file_id": str}).dropna(subset=["q_text"])
g = q[q["subject"] == "Computer Networks Lab"].reset_index(drop=True)
print("questions:", len(g), "| papers:", g["file_id"].nunique(),
      "| median words:", int(g["q_text"].str.split().str.len().median()) if len(g) else 0)
if len(g) > 5:
    S = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(g["q_text"]))
    best = []
    for i in range(len(g)):
        other = (g["file_id"] != g.loc[i, "file_id"]).to_numpy()
        best.append(S[i, other].max() if other.any() else 0)
    s = pd.Series(best)
    print(f"share with a near-copy (cosine>=0.7) in ANOTHER paper: {(s >= 0.7).mean():.0%}")
    print("\nsample questions:")
    for t in g["q_text"].sample(min(6, len(g)), random_state=1):
        print(" *", t[:160].replace("\n", " "))