"""Step 7 - FEATURES: encoding, scaling and normalization, only where they make sense.

Reads : data/processed/papers_clean.csv
Saves : data/processed/features.csv          (model-ready numeric table, one row per paper)
        data/processed/feature_strategy.csv  (documentation: what we did to each feature and why)

What we do (and what we deliberately do NOT do):
  ENCODING (categorical -> numbers)
    assessment_type, term  -> one-hot (few categories, no order between them)
    subject                -> NOT one-hot (131 values, many with only a few papers).
                              We keep the name for grouping and add subject_paper_count
                              (frequency encoding). Better encodings belong to Phase 3.
    yes/no flags           -> 0/1
  MISSING VALUES
    numeric text-derived features -> median of the same assessment type + a *_missing flag
    year                          -> left missing (we cannot honestly guess a year) + flag
  SCALING / NORMALIZATION (numeric only)
    skewed counts (|skewness| > 1) -> log1p first, so a few huge papers do not dominate
    counts                         -> z-score (mean 0, standard deviation 1)
    year                           -> min-max to 0..1 (a bounded "recency" value)
  TEXT: no normalization or scaling now. Question text is cleaned/vectorised in Phase 3.
Note: in Phase 3 the medians/means/std must be learned on the TRAINING split only.
"""
import numpy as np
import pandas as pd

from common import PROCESSED_DIR, log

LOG = "features_log.txt"
SKEW_LIMIT = 1.0
COUNT_COLUMNS = ["page_count", "word_count", "total_marks", "question_count_est",
                 "subject_paper_count"]
FLAG_COLUMNS = ["is_solution", "is_retake", "is_lab_subject", "has_text", "outlier_any"]


def one_hot(series, prefix):
    """One 0/1 column per category."""
    return pd.get_dummies(series, prefix=prefix, dtype=int)


def zscore(series):
    """(x - mean) / std, using the population standard deviation (like scikit-learn)."""
    std = series.std(ddof=0)
    return (series - series.mean()) / std if std > 0 else series * 0.0


def build_features(clean):
    """Return (features table, strategy table, scaling check table)."""
    df = clean.copy()
    df["term"] = df["term"].fillna("Unknown")
    df["subject"] = df["subject"].fillna("Unknown")
    df["subject_paper_count"] = df.groupby("subject")["file_id"].transform("count")
    # 0 words only means "no text", so it is missing information, not a short paper
    df.loc[~df["has_text"], "word_count"] = np.nan

    features = df[["file_id", "subject", "assessment_type"]].copy()  # kept for grouping/reference
    features = pd.concat([features, one_hot(df["assessment_type"], "type"),
                          one_hot(df["term"], "term")], axis=1)
    for column in FLAG_COLUMNS:
        features[column] = df[column].astype(int)

    # Missing indicators first (before we fill anything)
    features["year_missing"] = df["year"].isna().astype(int)
    for column in ["page_count", "word_count", "total_marks", "question_count_est"]:
        features[f"{column}_missing"] = df[column].isna().astype(int)

    # Year -> recency in 0..1 (stays missing when the year is unknown)
    year_min, year_max = df["year"].min(), df["year"].max()
    features["year_scaled"] = (df["year"] - year_min) / (year_max - year_min)

    # Counts: fill with median of the same assessment type, log1p if skewed, then z-score
    notes, checks = {}, []
    for column in COUNT_COLUMNS:
        values = df[column].copy()
        medians = values.groupby(df["assessment_type"]).transform("median")
        filled = values.fillna(medians).fillna(values.median())
        skew = float(filled.skew())
        use_log = abs(skew) > SKEW_LIMIT
        transformed = np.log1p(filled) if use_log else filled
        features[f"{column}_z"] = zscore(transformed)
        notes[column] = (round(skew, 2), use_log, int(values.isna().sum()))
        checks.append({"feature": f"{column}_z", "mean_after": round(features[f'{column}_z'].mean(), 3),
                       "std_after": round(features[f'{column}_z'].std(ddof=0), 3)})

    strategy = build_strategy_table(notes, year_min, year_max)
    return features, strategy, pd.DataFrame(checks)


def build_strategy_table(notes, year_min, year_max):
    """The documentation table that goes into the report."""
    rows = [
        ("assessment_type", "categorical (nominal)", "one-hot", "no", "no",
         "few categories, no natural order"),
        ("term (Spring/Fall/Summer/Unknown)", "categorical (nominal)", "one-hot", "no", "no",
         "few categories; 'Unknown' kept as its own category instead of dropping rows"),
        ("subject", "categorical (high cardinality)", "NOT one-hot; frequency count kept", "no", "no",
         "too many values with few papers each; richer encoding deferred to Phase 3"),
        ("is_solution, is_retake, is_lab_subject, has_text, outlier_any", "binary", "0/1", "no", "no",
         "already two-valued"),
        ("year", "numeric (time)", "no", f"min-max to 0..1 ({int(year_min)}-{int(year_max)})", "no",
         "bounded value used as recency; unknown years stay missing + year_missing flag"),
    ]
    for column, (skew, used_log, n_missing) in notes.items():
        rows.append((column, "numeric (count)", "no",
                     "z-score" + (" after log1p" if used_log else ""), "log1p" if used_log else "no",
                     f"skewness {skew}; {n_missing} missing filled with median of assessment type "
                     f"+ {column}_missing flag" if column != "subject_paper_count"
                     else f"skewness {skew}; derived from subject (frequency encoding)"))
    rows.append(("question text / paper text", "text", "no (Phase 3)", "no", "not applicable",
                 "numeric scaling is meaningless for text; vectorisation happens in Phase 3"))
    return pd.DataFrame(rows, columns=["feature", "type", "encoding", "scaling",
                                       "normalization / transform", "reason"])


def main():
    clean = pd.read_csv(PROCESSED_DIR / "papers_clean.csv")
    log(f"Building features for {len(clean)} clean papers", LOG)
    features, strategy, checks = build_features(clean)
    features.to_csv(PROCESSED_DIR / "features.csv", index=False, encoding="utf-8")
    strategy.to_csv(PROCESSED_DIR / "feature_strategy.csv", index=False, encoding="utf-8")

    log(f"features.csv: {features.shape[0]} rows x {features.shape[1]} columns", LOG)
    log(f"One-hot columns: {[c for c in features.columns if c.startswith(('type_', 'term_'))]}", LOG)
    for row in strategy.itertuples():
        if row.type == "numeric (count)":
            log(f"  {row.feature}: {row.scaling}; {row.reason}", LOG)
    log("Scaling check (should be mean ~0, std ~1):", LOG)
    for row in checks.itertuples():
        log(f"  {row.feature}: mean={row.mean_after}, std={row.std_after}", LOG)
    log(f"year_missing rows (kept missing): {int(features['year_missing'].sum())}", LOG)
    log(f"Saved {PROCESSED_DIR / 'features.csv'} and feature_strategy.csv", LOG)


if __name__ == "__main__":
    main()