"""Inference layer tests. They need the derived data (data/interim/...) and the
model files (models/...), so they are skipped on a clone that does not have them."""
import numpy as np
import pandas as pd
import pytest

from src.inference import config as C
from src.inference.errors import LowConfidenceError

needs_data = pytest.mark.skipif(
    not (C.QUESTIONS_CSV.exists() and (C.MODELS_DIR / "preprocess.joblib").exists()),
    reason="derived data or model files not present")

FEATURES = ["prior_papers", "prior_years", "topic_count_prior", "topic_rate_prior",
            "years_since_last", "appeared_last_year", "recent_rate", "type_rate_prior"]


@pytest.fixture(scope="module")
def pred():
    from src.inference.predictor import Predictor
    return Predictor()


@needs_data
def test_feature_parity_with_training_data(pred):
    """Serving features == training features on every row of train/val/test."""
    from src.inference.features import build_rows
    from src.models.preprocess import load_split
    bad = []
    for split in ("train", "val", "test"):
        ds = load_split(split)
        for (subj, yr, typ), g in ds.groupby(["subject", "target_year", "target_type"]):
            mine = build_rows(pred.papers, pred.incidence, g["topic_id"], subj, int(yr), typ)
            m = g.merge(mine, on="topic_id", suffixes=("_ds", "_me"))
            for c in FEATURES:
                if not np.allclose(m[c + "_ds"], m[c + "_me"], equal_nan=True, atol=1e-4):
                    bad.append((split, subj, yr, typ, c))
    assert not bad, bad[:5]


@needs_data
@pytest.mark.parametrize("model", ["logreg", "random_forest", "mlp"])
def test_scores_match_committed_predictions(pred, model):
    """The app reproduces the probabilities reported in Deliverable 2."""
    if model == "mlp":
        pytest.importorskip("torch")
    ref = pd.read_csv(C.PRED_DIR / f"{model}_test.csv")
    for (subj, yr, typ), g in list(ref.groupby(["subject", "target_year", "target_type"]))[:15]:
        out = pred.predict(subj, typ, int(yr), model=model)
        m = g.merge(out, on="topic_id")
        assert len(m) == len(g)
        assert np.allclose(m["score"], m["probability"], atol=1e-6)


@needs_data
def test_low_confidence_subject_raises(pred):
    with pytest.raises(LowConfidenceError):
        pred.predict(pred.low_confidence_subjects()[0], "mid", 2026)


@needs_data
def test_output_contract(pred):
    df = pred.predict(pred.subjects()[0], "final", 2027, model="logreg")
    assert df["probability"].between(0, 1).all()
    assert df["probability"].is_monotonic_decreasing
    assert set(df["band"]) <= {"Likely", "Possible", "Unlikely"}
    assert {"prior_papers", "threshold"} <= set(df.attrs)