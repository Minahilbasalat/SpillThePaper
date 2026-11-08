"""Scikit-Learn baselines: LogisticRegression and RandomForestClassifier.

Hyperparameters are chosen on VALIDATION AUC-ROC only. The test split is
never loaded in this file.

Run:  python -m src.models.baselines
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.models.preprocess import (
    ROOT,
    fit_preprocessor,
    load_split,
    save_preprocessor,
    split_to_xy,
)

SEED = 42
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

LOGREG_GRID = {"C": [0.01, 0.1, 1, 10]}
RF_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [1, 5],
}


def make_logreg(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000, random_state=SEED)


def make_rf(n_estimators: int, max_depth, min_samples_leaf: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=SEED,
        n_jobs=1,
    )


def grid_points(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def tune(name: str, factory, grid: dict, Xtr, ytr, Xva, yva):
    """Fit one model per grid point; return (rows, best_params, best_model)."""
    rows, best_auc, best_params, best_model = [], -1.0, None, None
    for params in grid_points(grid):
        model = factory(**params).fit(Xtr, ytr)
        val_auc = roc_auc_score(yva, model.predict_proba(Xva)[:, 1])
        train_auc = roc_auc_score(ytr, model.predict_proba(Xtr)[:, 1])
        rows.append(
            {"model": name, **params, "train_auc": train_auc, "val_auc": val_auc}
        )
        if val_auc > best_auc:
            best_auc, best_params, best_model = val_auc, params, model
    return rows, best_params, best_model


def main() -> None:
    train, val = load_split("train"), load_split("val")
    pre = fit_preprocessor(train)  # fitted on train only
    save_preprocessor(pre)
    Xtr, ytr = split_to_xy(train, pre)
    Xva, yva = split_to_xy(val, pre)

    # honest reference: one raw feature, no learning at all
    ref_auc = roc_auc_score(yva, val["topic_rate_prior"])
    print(f"Reference: raw topic_rate_prior, val AUC = {ref_auc:.3f}")

    rows, best = [], {}
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    for name, factory, grid in (
        ("logreg", make_logreg, LOGREG_GRID),
        ("random_forest", make_rf, RF_GRID),
    ):
        r, params, model = tune(name, factory, grid, Xtr, ytr, Xva, yva)
        rows += r
        best_auc = max(x["val_auc"] for x in r)
        best[name] = {"params": params, "val_auc": best_auc}
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"{name}: best val AUC = {best_auc:.3f} with {params}")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "baseline_tuning.csv", index=False)
    best["reference_topic_rate_prior_val_auc"] = ref_auc
    (RESULTS_DIR / "baseline_best.json").write_text(
        json.dumps(best, indent=2, default=str)
    )
    print(table.to_string(index=False))
    print("Saved results/baseline_tuning.csv and results/baseline_best.json")


if __name__ == "__main__":
    main()