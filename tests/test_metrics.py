import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    METRIC_NAMES,
    best_f1_threshold,
    compute_metrics,
    evaluate_with_ci,
    exam_group_indices,
    predict_labels,
)

# Hand-computed example (threshold 0.5): predictions are [1, 1, 1, 0, 0, 0].
Y = [1, 1, 0, 0, 1, 0]
S = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1]


def test_confusion_counts_and_precision_recall_f1():
    m = compute_metrics(Y, S, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_auc_pr_auc_and_brier_hand_values():
    m = compute_metrics(Y, S, 0.5)
    assert m["auc_roc"] == pytest.approx(7 / 9)  # 7 of 9 pos/neg pairs ranked right
    assert m["pr_auc"] == pytest.approx(13 / 15)  # (1 + 1 + 0.6) / 3
    assert m["brier"] == pytest.approx(0.2)  # 1.20 / 6


def test_threshold_is_inclusive():
    assert list(predict_labels([0.5, 0.2], 0.5)) == [1, 0]


def test_best_f1_threshold_hand_example():
    t, f1 = best_f1_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert t == pytest.approx(0.35)
    assert f1 == pytest.approx(0.8)


def test_single_class_gives_nan_auc():
    m = compute_metrics([1, 1, 1], [0.9, 0.5, 0.2], 0.5)
    assert np.isnan(m["auc_roc"]) and np.isnan(m["pr_auc"])


def test_brier_is_nan_for_scores_outside_unit_range():
    m = compute_metrics([1, 0, 1], [5.0, 0.0, 3.0], 2.0)
    assert np.isnan(m["brier"])


def _make_df(n_exams=40, rows=8, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for e in range(n_exams):
        label = rng.integers(0, 2, size=rows)
        score = np.clip(0.5 * label + rng.normal(0.25, 0.2, size=rows), 0, 1)
        parts.append(
            pd.DataFrame(
                {
                    "subject": f"S{e % 4}",
                    "target_year": 2020 + e,
                    "target_type": "mid",
                    "label": label,
                    "score": score,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_exam_groups_are_whole_exams():
    groups = exam_group_indices(_make_df())
    assert len(groups) == 40
    assert all(len(g) == 8 for g in groups)


def test_bootstrap_interval_contains_point_estimate():
    res = evaluate_with_ci(_make_df(), 0.5, n_boot=200, seed=1)
    assert res["n_exams"] == 40
    for name in METRIC_NAMES:
        assert res[f"{name}_lo"] <= res[name] <= res[f"{name}_hi"], name
    assert res["auc_roc_lo"] < res["auc_roc_hi"]


def test_bootstrap_is_reproducible_with_fixed_seed():
    df = _make_df()
    a = evaluate_with_ci(df, 0.5, n_boot=100, seed=7)
    b = evaluate_with_ci(df, 0.5, n_boot=100, seed=7)
    assert a == b