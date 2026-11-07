"""Create DUMMY predictions so the evaluation code can be built and tested
before the real model predictions exist.

Scores are random and only weakly related to the label. NEVER report any
number computed from these files. Output goes to results/dummy/predictions/
(ignored by Git).

Run:  python -m src.evaluation.dummy_predictions
"""

from __future__ import annotations

import argparse

import numpy as np

from src.data.schema import DUMMY_DATA_DIR
from src.evaluation.predictions_io import (
    REPO_ROOT,
    make_predictions_frame,
    read_split,
    validate_predictions,
)

MODELS = {"logreg": 0.10, "random_forest": 0.14, "mlp": 0.08}  # weak signal strength
SPLITS = ["val", "test"]
OUT_DIR = REPO_ROOT / "results" / "dummy" / "predictions"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_no, split in enumerate(SPLITS):
        split_df = read_split(split, DUMMY_DATA_DIR)
        y = split_df["label"].to_numpy(dtype=int)
        for model_no, (model, strength) in enumerate(MODELS.items()):
            rng = np.random.default_rng(args.seed + 100 * split_no + model_no)
            scores = np.clip(
                0.40 + strength * y + rng.normal(0.0, 0.20, size=len(y)), 0.001, 0.999
            )
            pred = make_predictions_frame(split_df, scores, model, split)
            validate_predictions(pred, split_df, model, split)
            path = OUT_DIR / f"{model}_{split}.csv"
            pred.to_csv(path, index=False)
            print(f"wrote {path.relative_to(REPO_ROOT)}  ({len(pred)} rows)")
    print("DUMMY data only - never report these numbers.")


if __name__ == "__main__":
    main()