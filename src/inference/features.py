"""Features of one upcoming exam, computed exactly like src/data/build_dataset.py.

Any change here must keep tests/test_inference.py::test_feature_parity passing,
otherwise the app would feed the models different numbers than they were trained on.
"""
import numpy as np
import pandas as pd

from src.inference.config import RECENT_WINDOW


def build_rows(papers: pd.DataFrame, incidence: pd.DataFrame, topic_ids,
               subject: str, target_year: int, target_type: str) -> pd.DataFrame:
    """papers:    file_id, subject, year, assessment_type (one row per paper)
    incidence: file_id, topic_id (the paper has at least one question in the topic)
    Only papers with year < target_year are used (no leakage)."""
    p = papers[(papers["subject"] == subject) & (papers["year"] < target_year)]
    n_prior, n_years = len(p), int(p["year"].nunique())
    same_type = p[p["assessment_type"] == target_type]
    recent = p[p["year"] >= target_year - RECENT_WINDOW]

    rows = []
    for tid in topic_ids:
        has = set(incidence.loc[incidence["topic_id"] == tid, "file_id"])
        hit = p[p["file_id"].isin(has)]
        cnt = len(hit)
        rate = cnt / n_prior if n_prior else 0.0
        type_rate = same_type["file_id"].isin(has).mean() if len(same_type) else rate
        rows.append({
            "topic_id": int(tid),
            "prior_papers": n_prior,
            "prior_years": n_years,
            "topic_count_prior": cnt,
            "topic_rate_prior": round(float(rate), 4),
            "years_since_last": float(target_year - hit["year"].max()) if cnt else np.nan,
            "appeared_last_year": int((hit["year"] == target_year - 1).any()),
            "recent_rate": round(float(recent["file_id"].isin(has).mean()), 4) if len(recent) else 0.0,
            "type_rate_prior": round(float(type_rate), 4),
            "target_type": target_type,
        })
    return pd.DataFrame(rows)