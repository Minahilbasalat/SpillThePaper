"""Simple non-learned baselines, written in the shared predictions format.

They show whether the learned models beat simple rules:
- rule_count_ge5   : "appeared 5 times, so predict it" (Deliverable 1 rule)
- rule_last_year   : predict 1 if the topic appeared in the previous year
- topic_rate_only  : raw topic_rate_prior used as the score
- all_positive     : predict 1 for every row

Run:  python -m src.evaluation.baselines            (real data -> results/predictions)
      python -m src.evaluation.baselines --dummy    (dummy data -> results/dummy/predictions)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.schema import DUMMY_DATA_DIR, MODEL_DATA_DIR
from src.evaluation.predictions_io import (
    REPO_ROOT,
    make_predictions_frame,
    read_split,
    validate_predictions,
)

SPLITS = ["val", "test"]
BASELINES = ["rule_count_ge5", "rule_last_year", "topic_rate_only", "all_positive"]

# Rules with a fixed decision threshold (score >= threshold -> predict 1).
# topic_rate_only is NOT listed: its threshold is chosen on validation.
FIXED_THRESHOLDS = {
    "rule_count_ge5": 5.0,
    "rule_last_year": 1.0,
    "all_positive": 1.0,
}

# Baselines whose score is a proper value in [0, 1].
UNIT_RANGE = {"topic_rate_only", "all_positive"}


def baseline_scores(df: pd.DataFrame, name: str) -> np.ndarray:
    """Score of one baseline for every row of a split."""
    count = df["topic_count_prior"].fillna(0).to_numpy(dtype=float)
    rate = df["topic_rate_prior"].fillna(0).to_numpy(dtype=float)
    last = df["appeared_last_year"].fillna(0).to_numpy(dtype=float)
    if name == "rule_count_ge5":
        return count
    if name == "rule_last_year":
        return last + 0.001 * rate  # tie-breaker never reaches 1
    if name == "topic_rate_only":
        return rate
    if name == "all_positive":
        return np.ones(len(df))
    raise ValueError(f"unknown baseline: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true", help="use dummy data")
    args = parser.parse_args()

    data_dir = DUMMY_DATA_DIR if args.dummy else MODEL_DATA_DIR
    out_dir = REPO_ROOT / "results" / ("dummy/predictions" if args.dummy else "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        df = read_split(split, data_dir)
        missing = {
            col: int(df[col].isna().sum())
            for col in ["topic_count_prior", "topic_rate_prior", "appeared_last_year"]
        }
        print(f"[{split}] missing feature values (filled with 0): {missing}")
        for name in BASELINES:
            scores = baseline_scores(df, name)
            pred = make_predictions_frame(df, scores, name, split)
            validate_predictions(
                pred, df, name, split, require_unit_range=name in UNIT_RANGE
            )
            path = out_dir / f"{name}_{split}.csv"
            pred.to_csv(path, index=False)
            print(f"wrote {path.relative_to(REPO_ROOT)}  ({len(pred)} rows)")


if __name__ == "__main__":
    main()