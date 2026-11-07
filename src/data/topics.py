"""Per-subject topic discovery on question text (unsupervised).

Topics are fitted on TRAIN-period questions only (year <= TRAIN_MAX_YEAR); later
questions are assigned to the nearest existing topic. This avoids using future
papers to define topics.

Outputs:
  data/interim/questions_topics.csv       question text + topic_id (NOT in Git)
  data/processed/topic_catalog.csv        subject, topic_id, keywords, counts
  data/processed/subject_eligibility.csv  which subjects get topics, and why not
"""
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.data.schema import TRAIN_MAX_YEAR

ROOT = Path(__file__).resolve().parents[2]
IN_QUESTIONS = ROOT / "data" / "interim" / "questions.csv"
OUT_QUESTIONS = ROOT / "data" / "interim" / "questions_topics.csv"
OUT_CATALOG = ROOT / "data" / "processed" / "topic_catalog.csv"
OUT_ELIGIBILITY = ROOT / "data" / "processed" / "subject_eligibility.csv"

# ---- settings (these go in the report's settings table) ----
MIN_TRAIN_PAPERS = 10      # a subject needs this many train-period papers ...
MIN_TRAIN_YEARS = 3        # ... from at least this many distinct years ...
MIN_TRAIN_QUESTIONS = 40   # ... and this many train-period questions
K_MIN, K_MAX = 4, 10       # range of topic counts tried per subject (silhouette)
QUESTIONS_PER_TOPIC = 10   # K is also capped at n_train_questions // this
TFIDF_MAX_FEATURES = 2000
TFIDF_NGRAM = (1, 2)
TFIDF_MIN_DF = 2
TOP_TERMS = 4
SEED = 42

EXAM_STOPWORDS = {
    "question", "questions", "marks", "mark", "clo", "write", "explain", "following",
    "given", "consider", "solution", "answer", "answers", "ans", "using", "use", "used",
    "show", "state", "define", "describe", "discuss", "list", "find", "draw", "mid",
    "final", "quiz", "points", "point", "part", "page", "total", "time", "attempt",
    "assume", "also", "case", "shown", "below", "above", "figure", "fig", "each",
    "let", "need", "must", "will", "would", "may", "one", "two", "three", "following",
}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | EXAM_STOPWORDS)
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z+#]{1,}\b"   # keeps tokens like c++ and c#


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True, stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN,
        ngram_range=TFIDF_NGRAM, min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES, sublinear_tf=True,
    )


def fit_subject_topics(train_texts):
    """Fit TF-IDF + KMeans; choose K by silhouette. Returns (vec, km, k, score)."""
    vec = make_vectorizer()
    X = vec.fit_transform(train_texts)
    if X.shape[1] < 10:
        return None
    k_max = min(K_MAX, len(train_texts) // QUESTIONS_PER_TOPIC)
    best = None
    for k in range(K_MIN, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(X, km.labels_, metric="cosine")
        if best is None or score > best[3]:
            best = (vec, km, k, float(score))
    return best


def topic_keywords(vec, km):
    terms = vec.get_feature_names_out()
    return {c: ", ".join(terms[i] for i in km.cluster_centers_[c].argsort()[::-1][:TOP_TERMS])
            for c in range(km.n_clusters)}


def main():
    q = pd.read_csv(IN_QUESTIONS, dtype={"file_id": str}).dropna(subset=["q_text"])
    q["year"] = q["year"].astype(int)

    eligibility, catalog, assigned = [], [], []
    for subject, grp in q.groupby("subject"):
        train = grp[grp["year"] <= TRAIN_MAX_YEAR]
        row = {
            "subject": subject,
            "train_papers": train["file_id"].nunique(),
            "train_years": train["year"].nunique(),
            "train_questions": len(train),
            "later_papers": grp[grp["year"] > TRAIN_MAX_YEAR]["file_id"].nunique(),
            "eligible": False, "reason": "", "k": None, "silhouette": None,
        }
        if row["train_papers"] < MIN_TRAIN_PAPERS:
            row["reason"] = f"fewer than {MIN_TRAIN_PAPERS} train papers"
        elif row["train_years"] < MIN_TRAIN_YEARS:
            row["reason"] = f"fewer than {MIN_TRAIN_YEARS} train years"
        elif row["train_questions"] < MIN_TRAIN_QUESTIONS:
            row["reason"] = f"fewer than {MIN_TRAIN_QUESTIONS} train questions"
        else:
            fit = fit_subject_topics(train["q_text"].tolist())
            if fit is None:
                row["reason"] = "vocabulary too small"
            else:
                vec, km, k, score = fit
                row.update(eligible=True, k=k, silhouette=round(score, 4))
                kw = topic_keywords(vec, km)
                g = grp.copy()
                g["topic_id"] = km.predict(vec.transform(g["q_text"]))
                assigned.append(g)
                train_topics = g[g["year"] <= TRAIN_MAX_YEAR]["topic_id"].value_counts()
                for t in range(k):
                    catalog.append({
                        "subject": subject, "topic_id": t, "keywords": kw[t],
                        "train_questions": int(train_topics.get(t, 0)),
                        "all_questions": int((g["topic_id"] == t).sum()),
                    })
        eligibility.append(row)

    elig = pd.DataFrame(eligibility).sort_values("train_papers", ascending=False)
    OUT_ELIGIBILITY.parent.mkdir(parents=True, exist_ok=True)
    elig.to_csv(OUT_ELIGIBILITY, index=False)
    pd.DataFrame(catalog).to_csv(OUT_CATALOG, index=False)
    out = pd.concat(assigned, ignore_index=True)
    out.to_csv(OUT_QUESTIONS, index=False)

    e = elig[elig["eligible"]]
    print(f"subjects total: {len(elig)} | eligible: {len(e)} | excluded: {len(elig) - len(e)}")
    print(f"topics created: {len(catalog)} | questions assigned: {len(out)} "
          f"of {len(q)} ({100 * len(out) / len(q):.1f}%)")
    print("chosen K (min/median/max):", int(e['k'].min()), float(e['k'].median()), int(e['k'].max()))
    print("silhouette (min/median/max):", e['silhouette'].min(), e['silhouette'].median(), e['silhouette'].max())
    print("papers after the train period in eligible subjects:", int(e["later_papers"].sum()))
    print("exclusion reasons:")
    print(elig[~elig["eligible"]]["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()