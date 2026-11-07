import numpy as np
import pandas as pd

from src.data.build_dataset import build_subject_rows

PAPERS = pd.DataFrame({
    "file_id": ["A", "B", "C", "D"],
    "year": [2018, 2019, 2019, 2021],
    "assessment_type": ["final", "final", "mid", "final"],
})


def _rows(presence):
    return pd.DataFrame(build_subject_rows("S", PAPERS, presence, ["t0", "t1"]))


def test_features_and_label_values():
    presence = np.array([[1, 0], [1, 0], [0, 0], [0, 1]], dtype=bool)
    df = _rows(presence)
    assert set(df["target_year"]) == {2021}          # earlier exams lack history
    r = df[df["topic_id"] == 0].iloc[0]
    assert r["prior_papers"] == 3
    assert r["topic_count_prior"] == 2
    assert abs(r["topic_rate_prior"] - 2 / 3) < 1e-3
    assert r["years_since_last"] == 2                # last seen 2019, target 2021
    assert r["appeared_last_year"] == 0              # nothing in 2020
    assert abs(r["recent_rate"] - 0.5) < 1e-9        # 2019 papers: B has it, C not
    assert r["type_rate_prior"] == 1.0               # A and B are finals, both have it
    assert r["label"] == 0
    assert df[df["topic_id"] == 1].iloc[0]["label"] == 1


def test_target_exam_never_changes_features():
    base = np.array([[1, 0], [1, 0], [0, 0], [0, 1]], dtype=bool)
    flipped = base.copy()
    flipped[3] = [1, 0]                              # change ONLY the target paper D
    a, b = _rows(base), _rows(flipped)
    feats = [c for c in a.columns if c not in ("label", "topic_label")]
    pd.testing.assert_frame_equal(a[feats], b[feats])
    assert not a["label"].equals(b["label"])         # but the label does change