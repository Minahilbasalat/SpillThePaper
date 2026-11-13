import numpy as np
import pandas as pd


def build_rows(papers: pd.DataFrame, incidence: pd.DataFrame, topic_ids,
               subject: str, target_year: int, target_type: str) -> pd.DataFrame:
    """papers:    paper_id, subject, year, assessment_type (one row per paper)
    incidence: paper_id, topic_id (a paper contains the topic)
    Uses only papers from years strictly before target_year."""
    p = papers[(papers.subject == subject) & (papers.year < target_year)]
    n_papers, n_years = len(p), p.year.nunique()
    same_type = p[p.assessment_type == target_type]
    recent = p[p.year >= target_year - 2]

    rows = []
    for tid in topic_ids:
        with_t = set(incidence.loc[incidence.topic_id == tid, "paper_id"])
        hit = p[p.paper_id.isin(with_t)]
        cnt = len(hit)
        rows.append({
            "topic_id": tid,
            "prior_papers": n_papers,
            "prior_years": n_years,
            "topic_count_prior": cnt,
            "topic_rate_prior": cnt / n_papers if n_papers else 0.0,
            "years_since_last": target_year - hit.year.max() if cnt else np.nan,
            "appeared_last_year": int((hit.year == target_year - 1).any()),
            "recent_rate": recent.paper_id.isin(with_t).mean() if len(recent) else 0.0,
            "type_rate_prior": same_type.paper_id.isin(with_t).mean() if len(same_type) else 0.0,
            "target_type": target_type,
        })
    return pd.DataFrame(rows)