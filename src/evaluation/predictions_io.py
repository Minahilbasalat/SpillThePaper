"""Shared predictions format (the output contract between Khadija and Iman).

results/predictions/<model>_<split>.csv with the columns, in this order:
row_idx, subject, topic_id, target_year, target_type, label, score, model, split
Rows follow the order of data/processed/model/<split>.csv exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_COLUMNS = [
    "row_idx",
    "subject",
    "topic_id",
    "target_year",
    "target_type",
    "label",
    "score",
    "model",
    "split",
]
_ID_CHECK_COLUMNS = ["subject", "topic_id", "target_year", "target_type", "label"]


def read_split(split: str, data_dir: str) -> pd.DataFrame:
    """Read data/processed/<model or model_dummy>/<split>.csv."""
    return pd.read_csv(REPO_ROOT / data_dir / f"{split}.csv")


def make_predictions_frame(
    split_df: pd.DataFrame, scores, model: str, split: str
) -> pd.DataFrame:
    """Build a predictions table in the contract format."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(split_df):
        raise ValueError(f"{len(scores)} scores for {len(split_df)} rows")
    out = pd.DataFrame(
        {
            "row_idx": np.arange(len(split_df)),
            "subject": split_df["subject"].to_numpy(),
            "topic_id": split_df["topic_id"].to_numpy(),
            "target_year": split_df["target_year"].to_numpy(),
            "target_type": split_df["target_type"].to_numpy(),
            "label": split_df["label"].to_numpy(dtype=int),
            "score": scores,
            "model": model,
            "split": split,
        }
    )
    return out[PREDICTION_COLUMNS]


def validate_predictions(
    pred: pd.DataFrame,
    split_df: pd.DataFrame,
    model: str,
    split: str,
    require_unit_range: bool = True,
) -> None:
    """Raise ValueError if pred breaks the contract (never silently fixes it)."""
    problems = []
    if list(pred.columns) != PREDICTION_COLUMNS:
        problems.append(f"columns are {list(pred.columns)}")
    else:
        if len(pred) != len(split_df):
            problems.append(f"{len(pred)} rows, expected {len(split_df)}")
        else:
            if not (pred["row_idx"].to_numpy() == np.arange(len(split_df))).all():
                problems.append("row_idx is not 0..n-1 in order")
            for col in _ID_CHECK_COLUMNS:
                same = (
                    pred[col].astype(str).to_numpy()
                    == split_df[col].astype(str).to_numpy()
                )
                if not same.all():
                    problems.append(f"column '{col}' differs from the split file")
        score = pred["score"].to_numpy(dtype=float)
        if not np.isfinite(score).all():
            problems.append("score has missing or infinite values")
        elif require_unit_range and ((score < 0) | (score > 1)).any():
            problems.append("score outside [0, 1]")
        if not (pred["model"] == model).all():
            problems.append(f"model column is not always '{model}'")
        if not (pred["split"] == split).all():
            problems.append(f"split column is not always '{split}'")
    if problems:
        raise ValueError(f"{model}_{split}: " + "; ".join(problems))