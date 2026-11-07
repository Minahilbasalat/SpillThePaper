"""Generate a DUMMY dataset with the real schema so teammates can develop the
model and evaluation code before the real dataset exists.

The values are random. Do NOT report results from this data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import (ALL_COLUMNS, DUMMY_DATA_DIR, TARGET)

ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(42)

SUBJECTS = ["Web Programming", "Computer Networks",
            "Design and Analysis of Algorithms", "Object Oriented Programming",
            "Artificial Intelligence", "Database Systems"]
TYPES = ["quiz", "mid", "final"]


def make_rows(years, n):
    prior_papers = rng.integers(1, 40, n)
    prior_years = np.minimum(prior_papers, rng.integers(1, 9, n))
    topic_count_prior = rng.binomial(prior_papers, rng.uniform(0.05, 0.6, n))
    topic_rate_prior = topic_count_prior / prior_papers
    years_since_last = np.where(topic_count_prior > 0,
                                rng.integers(1, 6, n).astype(float), np.nan)
    appeared_last_year = ((topic_count_prior > 0) & (rng.random(n) < 0.5)).astype(int)
    recent_rate = np.clip(topic_rate_prior + rng.normal(0, 0.1, n), 0, 1)
    type_rate_prior = np.clip(topic_rate_prior + rng.normal(0, 0.15, n), 0, 1)
    logit = -2.0 + 3.0 * topic_rate_prior + 1.0 * appeared_last_year + 1.5 * recent_rate
    label = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    topic_id = rng.integers(0, 8, n)
    return pd.DataFrame({
        "subject": rng.choice(SUBJECTS, n),
        "topic_id": topic_id,
        "topic_label": [f"topic_{t}" for t in topic_id],
        "target_year": rng.choice(years, n),
        "target_type": rng.choice(TYPES, n),
        "prior_papers": prior_papers,
        "prior_years": prior_years,
        "topic_count_prior": topic_count_prior,
        "topic_rate_prior": topic_rate_prior.round(4),
        "years_since_last": years_since_last,
        "appeared_last_year": appeared_last_year,
        "recent_rate": recent_rate.round(4),
        "type_rate_prior": type_rate_prior.round(4),
        TARGET: label,
    })[ALL_COLUMNS]


if __name__ == "__main__":
    out = ROOT / DUMMY_DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    make_rows(list(range(2016, 2023)), 3000).to_csv(out / "train.csv", index=False)
    make_rows([2023], 500).to_csv(out / "val.csv", index=False)
    make_rows([2024, 2025], 800).to_csv(out / "test.csv", index=False)
    print("dummy files written to", out)