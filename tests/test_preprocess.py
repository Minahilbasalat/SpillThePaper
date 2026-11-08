import numpy as np
import pandas as pd
import pytest

from src.data.schema import NUMERIC_FEATURES, TARGET
from src.models.preprocess import (
    NEVER_SEEN,
    YSL,
    Preprocessor,
    fit_preprocessor,
    load_preprocessor,
    load_split,
    save_preprocessor,
    split_to_xy,
)

TYPES = ["mid", "final", "quiz"]


def make_df(n, seed, scale=1.0, nan_rows=(), types=TYPES):
    """Small synthetic split with the real column names."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "subject": ["S"] * n,
            "topic_id": np.arange(n),
            "topic_label": ["t"] * n,
            "target_year": [2020] * n,
            "target_type": [types[i % len(types)] for i in range(n)],
        }
    )
    for col in NUMERIC_FEATURES:
        df[col] = rng.random(n) * scale
    df[YSL] = rng.integers(0, 6, n).astype(float)
    for i in nan_rows:
        df.loc[i, YSL] = np.nan
    df[TARGET] = rng.integers(0, 2, n)
    return df


def test_imputation_uses_train_max_plus_one():
    train = make_df(50, 0, nan_rows=[3, 7])
    pre = fit_preprocessor(train)
    expected = train[YSL].max() + 1
    assert pre.ysl_fill == expected

    X = pre.transform(train)
    assert X[NEVER_SEEN].sum() == 2
    assert X.loc[[3, 7], NEVER_SEEN].tolist() == [1.0, 1.0]
    # undo the standardisation: the imputed value must equal train max + 1
    raw = X.loc[3, YSL] * pre.std[YSL] + pre.mean[YSL]
    assert raw == pytest.approx(expected)


def test_fitted_statistics_come_from_train_only():
    train = make_df(50, 0, nan_rows=[3])
    val = make_df(30, 1, scale=100.0, nan_rows=[0])
    val[YSL] = 999.0  # extreme values that must not leak into the fit
    pre = fit_preprocessor(train)
    before = pre.state_dict()

    pre.transform(val)  # transforming other data must not change the state
    assert pre.state_dict() == before
    assert pre.ysl_fill == train[YSL].max() + 1

    pooled = pd.concat([train, val])
    for col in NUMERIC_FEATURES:
        if col == YSL:
            continue
        assert pre.mean[col] == pytest.approx(train[col].mean())
        assert pre.mean[col] != pytest.approx(pooled[col].mean())


def test_no_nan_after_transform():
    train = make_df(50, 0, nan_rows=[1, 2])
    val = make_df(30, 1, nan_rows=[0, 5])
    pre = fit_preprocessor(train)
    for df in (train, val):
        X = pre.transform(df)
        assert not X.isna().any().any()
        assert np.isfinite(X.to_numpy()).all()


def test_one_hot_uses_train_categories():
    train = make_df(40, 0, types=["mid", "final"])
    val = make_df(30, 1, types=["mid", "quiz"])
    pre = fit_preprocessor(train)
    Xt, Xv = pre.transform(train), pre.transform(val)

    assert list(Xt.columns) == list(Xv.columns)
    assert "target_type=quiz" not in Xv.columns
    onehot = [c for c in Xv.columns if c.startswith("target_type=")]
    quiz_rows = val["target_type"] == "quiz"
    assert (Xv.loc[quiz_rows, onehot].sum(axis=1) == 0).all()  # unseen -> zeros
    assert (Xv.loc[~quiz_rows, onehot].sum(axis=1) == 1).all()


def test_feature_names_and_order():
    pre = fit_preprocessor(make_df(50, 0))
    expected = (
        list(NUMERIC_FEATURES)
        + [NEVER_SEEN]
        + ["target_type=final", "target_type=mid", "target_type=quiz"]
    )
    assert pre.feature_names == expected
    assert list(pre.transform(make_df(10, 3)).columns) == expected


def test_train_numeric_columns_are_standardised():
    train = make_df(200, 0, scale=5.0, nan_rows=[4, 9])
    X = fit_preprocessor(train).transform(train)[list(NUMERIC_FEATURES)]
    assert np.allclose(X.mean(), 0.0, atol=1e-9)
    assert np.allclose(X.std(ddof=0), 1.0, atol=1e-9)


def test_save_and_load_roundtrip(tmp_path):
    train = make_df(50, 0, nan_rows=[2])
    val = make_df(20, 1, nan_rows=[0])
    pre = fit_preprocessor(train)
    path = save_preprocessor(pre, tmp_path / "pre.joblib")
    loaded = load_preprocessor(path)
    pd.testing.assert_frame_equal(pre.transform(val), loaded.transform(val))


def test_unfitted_preprocessor_raises():
    with pytest.raises(RuntimeError):
        Preprocessor().transform(make_df(5, 0))


def test_real_train_and_val_splits():
    train, val = load_split("train"), load_split("val")
    pre = fit_preprocessor(train)
    for df in (train, val):
        X, y = split_to_xy(df, pre)
        assert X.shape == (len(df), 12)  # 8 numeric + never_seen + 3 one-hot
        assert not np.isnan(X).any()
        assert set(np.unique(y)) <= {0, 1}