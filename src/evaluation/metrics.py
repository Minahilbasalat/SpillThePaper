"""Metrics, threshold selection and cluster bootstrap for Phase 4.

Conventions used everywhere in src/evaluation:
- A model produces a score in [0, 1] (probability that the topic appears).
- A row is predicted 1 when score >= threshold (inclusive).
- The threshold is chosen ONCE on the validation set (best F1) and then
  applied unchanged to the test set.
- Rows of the same target exam (subject + target_year + target_type) are
  correlated, so confidence intervals resample whole exams, not rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EXAM_KEYS = ["subject", "target_year", "target_type"]
METRIC_NAMES = ["precision", "recall", "f1", "auc_roc", "pr_auc", "brier"]


def predict_labels(scores, threshold: float) -> np.ndarray:
    """Hard 0/1 predictions: 1 when score >= threshold."""
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)


def compute_metrics(y_true, scores, threshold: float) -> dict:
    """Precision, recall, F1, AUC-ROC, PR-AUC, Brier and confusion counts.

    - AUC-ROC and PR-AUC are NaN when only one class is present.
    - Brier is NaN when some score lies outside [0, 1] (rule scores such as
      raw counts are not probabilities, so Brier would be meaningless).
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = predict_labels(s, threshold)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    two_classes = len(np.unique(y)) == 2
    auc_roc = float(roc_auc_score(y, s)) if two_classes else float("nan")
    pr_auc = float(average_precision_score(y, s)) if two_classes else float("nan")

    in_unit_range = bool(((s >= 0.0) & (s <= 1.0)).all())
    brier = float(np.mean((s - y) ** 2)) if in_unit_range else float("nan")

    return {
        "threshold": float(threshold),
        "n_rows": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "brier": brier,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    """Threshold that maximises F1 (use on the VALIDATION set only).

    Every distinct score is tried as a threshold. Ties go to the smallest
    threshold, so the choice is deterministic. Returns (threshold, f1).
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    best_t, best_f1 = float(np.min(s)), -1.0
    for t in np.unique(s):
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def exam_group_indices(df: pd.DataFrame) -> list[np.ndarray]:
    """Positional row indices of each target exam (subject, year, type)."""
    d = df.reset_index(drop=True)
    return [np.asarray(ix) for ix in d.groupby(EXAM_KEYS, sort=True).indices.values()]


def cluster_bootstrap(
    df: pd.DataFrame,
    threshold: float,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CIs, resampling whole target exams with replacement.

    df needs the columns: subject, target_year, target_type, label, score.
    Returns {metric: (low, high)} for every name in METRIC_NAMES.
    """
    d = df.reset_index(drop=True)
    y = d["label"].to_numpy(dtype=int)
    s = d["score"].to_numpy(dtype=float)
    groups = exam_group_indices(d)
    rng = np.random.default_rng(seed)

    draws = {name: [] for name in METRIC_NAMES}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        m = compute_metrics(y[idx], s[idx], threshold)
        for name in METRIC_NAMES:
            draws[name].append(m[name])

    ci = {}
    for name, values in draws.items():
        arr = np.asarray(values, dtype=float)
        if np.isnan(arr).all():
            ci[name] = (float("nan"), float("nan"))
        else:
            ci[name] = (
                float(np.nanpercentile(arr, 100 * alpha / 2)),
                float(np.nanpercentile(arr, 100 * (1 - alpha / 2))),
            )
    return ci


def evaluate_with_ci(
    df: pd.DataFrame, threshold: float, n_boot: int = 1000, seed: int = 42
) -> dict:
    """Point metrics plus 95% cluster-bootstrap CIs, as one flat dict."""
    d = df.reset_index(drop=True)
    point = compute_metrics(d["label"], d["score"], threshold)
    ci = cluster_bootstrap(d, threshold, n_boot=n_boot, seed=seed)
    row = dict(point)
    row["n_exams"] = len(exam_group_indices(d))
    for name in METRIC_NAMES:
        row[f"{name}_lo"], row[f"{name}_hi"] = ci[name]
    return row