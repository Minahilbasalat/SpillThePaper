"""Step 6 - CLEAN: duplicates, invalid values and outliers.

Reads : data/processed/papers.csv, data/processed/collection_manifest.csv (for file hashes),
        data/interim/text/<file_id>.txt
Saves : data/processed/papers_flagged.csv   (ALL rows + why a row was excluded, for transparency)
        data/processed/papers_clean.csv     (only the rows we keep for analysis)
        data/processed/cleaning_summary.csv (rows flagged / remaining after every step)
        data/processed/outlier_bounds.csv   (the outlier limits that were used)

Rules, in this order (a row is excluded by the FIRST rule that matches):
  1. exact_duplicate     - same file content (SHA-256). Keep one copy.
  2. same_text_duplicate - same text after removing case/spaces/punctuation. Keep one copy.
  3. merged_compilation  - several papers merged in one file; it repeats the single papers.
  4. solution_companion  - a "solution" file when the same paper (same subject, type, number,
                           year, term, section) also exists without "solution". Same questions.
Then:
  - invalid values: total_marks outside 1-200 is treated as a parsing mistake -> set to missing.
  - outliers (IQR rule, per assessment type) on numeric columns are FLAGGED, never deleted.
Rows without text are NOT removed here: their metadata (subject, year, type) is still useful.
"""
import hashlib
import re

import numpy as np
import pandas as pd

from common import PROCESSED_DIR, ROOT, log

TEXT_DIR = ROOT / "data" / "interim" / "text"
MIN_TEXT_CHARS = 200          # shorter texts are too small to compare reliably
MARKS_MIN, MARKS_MAX = 1, 200  # believable range for the total marks of one paper
IQR_FACTOR = 1.5
MIN_GROUP_SIZE = 20           # need at least this many values to compute outlier limits
LOG = "cleaning_log.txt"


def normalize_text(text):
    """Lower-case and keep only letters and digits, so layout differences do not matter."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def mark_duplicates(df, group_column, reason):
    """Among rows still kept that share the same value in group_column, keep ONE and exclude
    the rest. The kept copy prefers a known year, then the shorter path."""
    candidates = df[(df["exclusion_reason"] == "") & (df[group_column] != "")]
    count = 0
    for _, group in candidates.groupby(group_column):
        if len(group) < 2:
            continue
        order = (group.assign(_no_year=group["year"].isna(), _length=group["repo_path"].str.len())
                 .sort_values(["_no_year", "_length", "repo_path"]))
        keeper = order.iloc[0]["file_id"]
        for index in order.index[1:]:
            df.at[index, "exclusion_reason"] = reason
            df.at[index, "duplicate_of"] = keeper
            count += 1
    return count


def mark_merged(df):
    mask = (df["exclusion_reason"] == "") & df["is_merged"]
    df.loc[mask, "exclusion_reason"] = "merged_compilation"
    return int(mask.sum())


def mark_solution_companions(df):
    """Exclude a solution file when the same paper also exists as a non-solution file."""
    kept = df[df["exclusion_reason"] == ""]
    needs_number = kept["assessment_type"].isin(["mid", "quiz"])
    eligible = kept[kept["year"].notna() & (~needs_number | kept["assessment_number"].notna())]
    key_columns = ["subject", "assessment_type", "assessment_number", "year", "term",
                   "section_raw", "is_retake"]
    keys = eligible[key_columns].fillna(-1).astype(str).agg("|".join, axis=1)
    count = 0
    for _, group in eligible.groupby(keys):
        plain = group[~group["is_solution"]]
        solutions = group[group["is_solution"]]
        if len(plain) and len(solutions):
            for index in solutions.index:
                df.at[index, "exclusion_reason"] = "solution_companion"
                df.at[index, "duplicate_of"] = plain.iloc[0]["file_id"]
                count += 1
    return count


def fix_invalid_marks(df):
    """Total marks outside the believable range are parsing mistakes -> missing."""
    invalid = df["total_marks"].notna() & ~df["total_marks"].between(MARKS_MIN, MARKS_MAX)
    df.loc[invalid, "total_marks"] = np.nan
    return int(invalid.sum())


def flag_outliers(df):
    """IQR rule per assessment type. Flags only; never deletes. Returns a bounds table."""
    kept = df[df["exclusion_reason"] == ""]
    columns = {"page_count": "outlier_pages", "word_count": "outlier_words",
               "total_marks": "outlier_marks", "question_count_est": "outlier_questions"}
    bounds = []
    for column, flag in columns.items():
        df[flag] = False
        values = df[column].copy()
        if column == "word_count":
            values = values.where(df["has_text"])  # 0 words just means "no text", not an outlier
        for assessment_type, group in kept.groupby("assessment_type"):
            group_values = values.loc[group.index].dropna()
            if len(group_values) < MIN_GROUP_SIZE:
                continue
            q1, q3 = group_values.quantile(0.25), group_values.quantile(0.75)
            low, high = q1 - IQR_FACTOR * (q3 - q1), q3 + IQR_FACTOR * (q3 - q1)
            is_outlier = group_values[(group_values < low) | (group_values > high)].index
            df.loc[is_outlier, flag] = True
            bounds.append({"assessment_type": assessment_type, "column": column,
                           "n_values": len(group_values), "lower": round(low, 1),
                           "upper": round(high, 1), "flagged": len(is_outlier)})
    df["outlier_any"] = df[list(columns.values())].any(axis=1)
    return pd.DataFrame(bounds)


def run_cleaning(papers, text_lookup):
    """Apply all rules to a papers table. text_lookup(file_id) returns the text or ''."""
    df = papers.copy()
    df["exclusion_reason"] = ""
    df["duplicate_of"] = ""
    summary = [("start", 0, len(df))]

    def step(name, count):
        summary.append((name, count, int((df["exclusion_reason"] == "").sum())))

    step("exact_duplicate", mark_duplicates(df, "sha256", "exact_duplicate"))

    df["text_hash"] = ""
    for row in df[df["has_text"]].itertuples():
        text = normalize_text(text_lookup(row.file_id))
        if len(text) >= MIN_TEXT_CHARS:
            df.at[row.Index, "text_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    step("same_text_duplicate", mark_duplicates(df, "text_hash", "same_text_duplicate"))
    step("merged_compilation", mark_merged(df))
    step("solution_companion", mark_solution_companions(df))

    invalid_marks = fix_invalid_marks(df)
    bounds = flag_outliers(df)
    return df, pd.DataFrame(summary, columns=["step", "rows_flagged", "rows_remaining"]), \
        bounds, invalid_marks


def main():
    papers = pd.read_csv(PROCESSED_DIR / "papers.csv", keep_default_na=False,
                         true_values=["True"], false_values=["False"])
    for column in ["year", "assessment_number", "total_marks", "question_count_est", "page_count"]:
        papers[column] = pd.to_numeric(papers[column], errors="coerce")
    for column in ["is_solution", "is_merged", "is_retake", "has_text"]:
        papers[column] = papers[column].astype(bool)
    manifest = pd.read_csv(PROCESSED_DIR / "collection_manifest.csv", keep_default_na=False)
    papers = papers.merge(manifest[["file_id", "sha256"]], on="file_id", how="left")
    papers["sha256"] = papers["sha256"].fillna("")

    def text_lookup(file_id):
        path = TEXT_DIR / f"{file_id}.txt"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    log(f"Cleaning {len(papers)} exam papers", LOG)
    df, summary, bounds, invalid_marks = run_cleaning(papers, text_lookup)

    clean = df[df["exclusion_reason"] == ""].drop(columns=["exclusion_reason", "duplicate_of", "text_hash"])
    df.drop(columns=["text_hash"]).to_csv(PROCESSED_DIR / "papers_flagged.csv", index=False, encoding="utf-8")
    clean.to_csv(PROCESSED_DIR / "papers_clean.csv", index=False, encoding="utf-8")
    summary.to_csv(PROCESSED_DIR / "cleaning_summary.csv", index=False, encoding="utf-8")
    bounds.to_csv(PROCESSED_DIR / "outlier_bounds.csv", index=False, encoding="utf-8")

    log("Rows after each step (flagged = newly excluded):", LOG)
    for r in summary.itertuples():
        log(f"  {r.step}: flagged {r.rows_flagged}, remaining {r.rows_remaining}", LOG)
    log(f"Invalid total_marks set to missing: {invalid_marks}", LOG)
    log(f"Outliers flagged (kept rows): pages={int(clean['outlier_pages'].sum())} "
        f"words={int(clean['outlier_words'].sum())} marks={int(clean['outlier_marks'].sum())} "
        f"questions={int(clean['outlier_questions'].sum())}  any={int(clean['outlier_any'].sum())}", LOG)
    log(f"Final clean table: {len(clean)} rows, {clean['subject'].nunique()} subjects", LOG)
    log(f"Clean rows with readable text: {int(clean['has_text'].sum())}", LOG)


if __name__ == "__main__":
    main()