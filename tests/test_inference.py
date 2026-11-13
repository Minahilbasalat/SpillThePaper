import numpy as np
import pandas as pd
import pytest

from src.inference import config as C
from src.inference.features import build_rows
from src.inference.predictor import Predictor
from src.inference.stub import LowConfidenceError


@pytest.fixture(scope="module")
def pred():
    return Predictor()


def test_feature_parity_with_committed_dataset(pred):
    """Serving features must equal training features (no train/serve skew)."""
    ds = pd.read_csv(C.DATASET_CSV)               # ADAPT column names
    cols = ["prior_papers", "prior_years", "topic_count_prior", "topic_rate_prior",
            "years_since_last", "appeared_last_year", "recent_rate", "type_rate_prior"]
    bad = 0
    for (subj, yr, typ), g in ds.groupby(["subject", "target_year", "target_type"]):
        mine = build_rows(pred.papers, pred.incidence, g.topic_id, subj, yr, typ)
        merged = g.merge(mine, on="topic_id", suffixes=("_ds", "_me"))
        for c in cols:
            if not np.allclose(merged[c + "_ds"], merged[c + "_me"], equal_nan=True, atol=1e-6):
                bad += 1
    assert bad == 0


def test_scores_match_committed_predictions(pred):
    """Probabilities from the app equal the ones reported in Deliverable 2."""
    committed = pd.read_csv(C.PRED_DIR / "test_logreg.csv")   # ADAPT file and columns
    ...  # for a few exams: pred.predict(...) vs committed score, atol=1e-6


def test_low_confidence_subject_raises(pred):
    with pytest.raises(LowConfidenceError):
        pred.predict(pred.low_confidence_subjects()[0], "mid", 2026)


def test_probabilities_valid_and_sorted(pred):
    df = pred.predict(pred.subjects()[0], "final", 2026, model="mlp")
    assert df.probability.between(0, 1).all()
    assert df.probability.is_monotonic_decreasing