"""Write prediction files following the shared output contract.

  results/predictions/<model>_<split>.csv
  columns: row_idx, subject, topic_id, target_year, target_type,
           label, score, model, split

The models and the preprocessor were fitted on train only. The test
split is used here only to score already-chosen models.

Run:  python -m src.models.export_predictions
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.mlp import MLP, predict_proba
from src.models.preprocess import (
    ROOT,
    load_preprocessor,
    load_split,
    split_to_xy,
)

PRED_DIR = ROOT / "results" / "predictions"
MODELS_DIR = ROOT / "models"
COLUMNS = [
    "row_idx", "subject", "topic_id", "target_year", "target_type",
    "label", "score", "model", "split",
]
EXPECTED_ROWS = {"val": 301, "test": 568}
SKLEARN_MODELS = ("logreg", "random_forest")


def build_frame(df: pd.DataFrame, scores: np.ndarray, model: str, split: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "row_idx": np.arange(len(df)),
            "subject": df["subject"].to_numpy(),
            "topic_id": df["topic_id"].to_numpy(),
            "target_year": df["target_year"].to_numpy(),
            "target_type": df["target_type"].to_numpy(),
            "label": df["label"].to_numpy(),
            "score": np.asarray(scores, dtype="float64"),
            "model": model,
            "split": split,
        }
    )
    return out[COLUMNS]


def check_contract(frame: pd.DataFrame, split: str) -> None:
    assert list(frame.columns) == COLUMNS, "wrong columns or order"
    assert len(frame) == EXPECTED_ROWS[split], f"{split}: wrong row count"
    assert frame["score"].between(0, 1).all(), "score outside [0, 1]"
    assert frame["score"].notna().all(), "NaN score"
    assert (frame["row_idx"].to_numpy() == np.arange(len(frame))).all()


def write_predictions(model_name: str, scores_by_split: dict[str, np.ndarray]) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for split, scores in scores_by_split.items():
        df = load_split(split)
        frame = build_frame(df, scores, model_name, split)
        check_contract(frame, split)
        path = PRED_DIR / f"{model_name}_{split}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path.name}: {len(frame)} rows")


def export_sklearn(model_name: str, pre) -> None:
    model = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    scores = {}
    for split in ("val", "test"):
        X, _ = split_to_xy(load_split(split), pre)
        scores[split] = model.predict_proba(X)[:, 1]
    write_predictions(model_name, scores)


def export_mlp(pre) -> None:
    ckpt = torch.load(MODELS_DIR / "mlp_best.pt")
    h1, h2 = ckpt["config"]["hidden"]
    model = MLP(ckpt["in_features"], h1, h2, ckpt["config"]["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    scores = {}
    for split in ("val", "test"):
        X, _ = split_to_xy(load_split(split), pre)
        scores[split] = predict_proba(model, X)
    write_predictions("mlp", scores)


def main() -> None:
    pre = load_preprocessor()
    for name in SKLEARN_MODELS:
        export_sklearn(name, pre)
    export_mlp(pre)


if __name__ == "__main__":
    main()