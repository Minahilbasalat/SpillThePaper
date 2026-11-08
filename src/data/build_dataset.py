"""Build the supervised dataset: will topic T appear in the (subject, year, type) exam?

Row = (subject, topic, target_year, target_type).
Label = 1 if any paper of that subject/year/type contains a question assigned to the topic.
Features use ONLY papers with year < target_year (no leakage).

Writes data/processed/model/{train,val,test}.csv and
data/processed/model_dataset_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import (ALL_COLUMNS, MODEL_DATA_DIR, TEST_MIN_YEAR,
                             TRAIN_MAX_YEAR, VAL_YEARS)

ROOT = Path(__file__).resolve().parents[2]
IN_QUESTIONS = ROOT / "data" / "interim" / "questions_topics.csv"
IN_CATALOG = ROOT / "data" / "processed" / "topic_catalog.csv"
OUT_DIR = ROOT / MODEL_DATA_DIR
OUT_SUMMARY = ROOT / "data" / "processed" / "model_dataset_summary.csv"

# ---- settings (these go in the report's settings table) ----
MIN_PRIOR_PAPERS = 3   # a target exam needs this many earlier papers of the subject
MIN_PRIOR_YEARS = 2    # ... from at least this many distinct earlier years
RECENT_WINDOW = 2      # "recent" = the last 2 years before the target year


def build_subject_rows(subject, papers, presence, keywords):
    """papers: DataFrame [file_id, year, assessment_type], one row per paper.
    presence: bool array (n_papers, k): does paper i contain topic t?
    keywords: list of k topic keyword strings.
    """
    papers = papers.reset_index(drop=True)
    years = papers["year"].to_numpy()
    types = papers["assessment_type"].to_numpy()
    k = presence.shape[1]
    rows = []
    groups = papers.groupby(["year", "assessment_type"]).groups

    for (ty, tt), idx in sorted(groups.items()):
        ty = int(ty)
        idx = np.asarray(idx)
        prior = years < ty
        n_prior = int(prior.sum())
        n_years = len(set(years[prior].tolist()))
        if n_prior < MIN_PRIOR_PAPERS or n_years < MIN_PRIOR_YEARS:
            continue

        label = presence[idx].any(axis=0)              # the only use of target-year papers
        P = presence[prior]
        count = P.sum(axis=0)
        rate = count / n_prior
        prior_year_values = years[prior]
        last_year = np.array([prior_year_values[P[:, t]].max() if count[t] else np.nan
                              for t in range(k)])
        since = ty - last_year                         # NaN when never seen before

        last = years == ty - 1
        appeared_last = presence[last].any(axis=0).astype(int) if last.any() else np.zeros(k, int)
        recent = prior & (years >= ty - RECENT_WINDOW)
        recent_rate = presence[recent].mean(axis=0) if recent.any() else np.zeros(k)
        same_type = prior & (types == tt)
        type_rate = presence[same_type].mean(axis=0) if same_type.any() else rate

        for t in range(k):
            rows.append({
                "subject": subject, "topic_id": t, "topic_label": keywords[t],
                "target_year": ty, "target_type": tt,
                "prior_papers": n_prior, "prior_years": n_years,
                "topic_count_prior": int(count[t]),
                "topic_rate_prior": round(float(rate[t]), 4),
                "years_since_last": since[t],
                "appeared_last_year": int(appeared_last[t]),
                "recent_rate": round(float(recent_rate[t]), 4),
                "type_rate_prior": round(float(type_rate[t]), 4),
                "label": int(label[t]),
            })
    return rows


def split_by_year(df):
    train = df[df["target_year"] <= TRAIN_MAX_YEAR]
    val = df[df["target_year"].isin(VAL_YEARS)]
    test = df[df["target_year"] >= TEST_MIN_YEAR]
    return train, val, test


def summarise(splits):
    out = []
    for name, d in splits.items():
        out.append({
            "split": name, "rows": len(d), "positives": int(d["label"].sum()),
            "positive_rate": round(float(d["label"].mean()), 4) if len(d) else None,
            "subjects": d["subject"].nunique(),
            "target_exams": len(d[["subject", "target_year", "target_type"]].drop_duplicates()),
            "min_year": int(d["target_year"].min()) if len(d) else None,
            "max_year": int(d["target_year"].max()) if len(d) else None,
        })
    return pd.DataFrame(out)


def main():
    q = pd.read_csv(IN_QUESTIONS, dtype={"file_id": str})
    cat = pd.read_csv(IN_CATALOG)
    q["year"] = q["year"].astype(int)

    all_rows = []
    for subject, grp in q.groupby("subject"):
        keywords = cat[cat["subject"] == subject].sort_values("topic_id")["keywords"].tolist()
        k = len(keywords)
        papers = (grp[["file_id", "year", "assessment_type"]]
                  .drop_duplicates("file_id").reset_index(drop=True))
        pos = {f: i for i, f in enumerate(papers["file_id"])}
        presence = np.zeros((len(papers), k), dtype=bool)
        for fid, t in grp[["file_id", "topic_id"]].drop_duplicates().itertuples(index=False):
            presence[pos[fid], int(t)] = True
        all_rows += build_subject_rows(subject, papers, presence, keywords)

    df = pd.DataFrame(all_rows)[ALL_COLUMNS]
    train, val, test = split_by_year(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    summary = summarise({"train": train, "val": val, "test": test})
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))
    print("\npositive rate by target_type (all rows):")
    print(df.groupby("target_type")["label"].agg(["count", "mean"]).round(3).to_string())
    print("\nrows with no prior appearance of the topic:",
          f"{100 * (df['topic_count_prior'] == 0).mean():.1f}%")
    print("saved to", OUT_DIR)


if __name__ == "__main__":
    main()