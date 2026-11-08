"""Preprocessing for the SpillThePaper Phase 3 models.

Everything is FITTED ON THE TRAIN SPLIT ONLY:
  - years_since_last NaN (topic never seen before) -> train max + 1,
    plus a never_seen indicator column
  - target_type one-hot encoded with the categories seen in train
  - numeric features standardised with train mean / std

Identifier columns (subject, topic_id, topic_label) are NOT features.

The fitted state is saved as a plain dict (not a class instance) so that
joblib.load works no matter how this module was started.

Run:  python -m src.models.preprocess
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.schema import (
    CATEGORICAL_FEATURES,
    MODEL_DATA_DIR,
    NUMERIC_FEATURES,
    TARGET,
)

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
PREPROCESS_PATH = ROOT / "models" / "preprocess.joblib"
NEVER_SEEN = "never_seen"
YSL = "years_since_last"


def load_split(split: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Read data/processed/model/<split>.csv."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    folder = Path(data_dir) if data_dir else ROOT / MODEL_DATA_DIR
    return pd.read_csv(folder / f"{split}.csv")


class Preprocessor:
    """Train-only imputation, one-hot encoding and standardisation."""

    def __init__(self) -> None:
        self.fitted = False
        self.ysl_fill: float = float("nan")
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        ysl = train_df[YSL]
        if ysl.notna().sum() == 0:
            raise ValueError("years_since_last has no observed value in train")
        self.ysl_fill = float(ysl.max() + 1)

        filled = self._impute(train_df)[NUMERIC_FEATURES]
        self.mean = {c: float(filled[c].mean()) for c in NUMERIC_FEATURES}
        self.std = {}
        for c in NUMERIC_FEATURES:
            s = float(filled[c].std(ddof=0))
            self.std[c] = s if s > 0 else 1.0  # constant column: avoid /0

        self.categories = {
            c: sorted(train_df[c].dropna().unique().tolist())
            for c in CATEGORICAL_FEATURES
        }
        self.fitted = True
        return self

    # ------------------------------------------------------------ transform
    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[NUMERIC_FEATURES].astype(float).copy()
        out[NEVER_SEEN] = df[YSL].isna().astype(float)
        out[YSL] = out[YSL].fillna(self.ysl_fill)
        return out

    @property
    def feature_names(self) -> list[str]:
        names = list(NUMERIC_FEATURES) + [NEVER_SEEN]
        for col in CATEGORICAL_FEATURES:
            names += [f"{col}={cat}" for cat in self.categories[col]]
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor is not fitted yet; call fit() first")
        out = self._impute(df)
        for c in NUMERIC_FEATURES:
            out[c] = (out[c] - self.mean[c]) / self.std[c]
        for col in CATEGORICAL_FEATURES:
            for cat in self.categories[col]:
                # a category never seen in train gives all zeros
                out[f"{col}={cat}"] = (df[col] == cat).astype(float)
        return out[self.feature_names]

    # ---------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {
            "ysl_fill": self.ysl_fill,
            "mean": dict(self.mean),
            "std": dict(self.std),
            "categories": {k: list(v) for k, v in self.categories.items()},
        }

    @classmethod
    def from_state(cls, state: dict) -> "Preprocessor":
        pre = cls()
        pre.ysl_fill = state["ysl_fill"]
        pre.mean = dict(state["mean"])
        pre.std = dict(state["std"])
        pre.categories = {k: list(v) for k, v in state["categories"].items()}
        pre.fitted = True
        return pre


def fit_preprocessor(train_df: pd.DataFrame) -> Preprocessor:
    return Preprocessor().fit(train_df)


def save_preprocessor(pre: Preprocessor, path: str | Path = PREPROCESS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre.state_dict(), path)
    return path


def load_preprocessor(path: str | Path = PREPROCESS_PATH) -> Preprocessor:
    return Preprocessor.from_state(joblib.load(path))


def split_to_xy(df: pd.DataFrame, pre: Preprocessor) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) as numpy arrays using an already-fitted preprocessor."""
    X = pre.transform(df).to_numpy(dtype="float64")
    y = df[TARGET].to_numpy(dtype="int64")
    return X, y


def main() -> None:
    train = load_split("train")
    pre = fit_preprocessor(train)
    path = save_preprocessor(pre)
    print(f"Fitted on train only ({len(train)} rows).")
    print(f"years_since_last fill value (train max + 1): {pre.ysl_fill}")
    print(f"Features ({len(pre.feature_names)}): {pre.feature_names}")
    for split in ("train", "val"):
        X, y = split_to_xy(load_split(split), pre)
        print(f"{split}: X{X.shape}, y{y.shape}, NaN in X: {int(np.isnan(X).sum())}")
    print(f"Saved preprocessor state to {path}")


if __name__ == "__main__":
    main()