"""Edge cases found while testing the backend with bad input."""
import numpy as np
import pytest

from src.inference import config as C
from src.inference.generator import fit_marks

needs_topics = pytest.mark.skipif(not (C.MODELS_DIR / "topics").exists(), reason="topic models missing")


@pytest.mark.parametrize("raw, total", [([5, 10, 15], 30), ([1] * 18, 20), ([3, 3, 3, 3], 10), ([10, 0.5, 0.5], 5),
                                        ([1.0] * 10, 7), ([7, 2], 100), ([5, 5, 5], 17.5), ([2.5, 40, 3, 3, 3, 8], 21)])
def test_marks_are_whole_positive_and_add_up(raw, total):
    marks, used = fit_marks(np.array(raw, dtype=float), total)
    assert (marks >= 1).all() and (marks == np.round(marks)).all()
    assert marks.sum() == used >= len(raw)
    assert used == max(int(round(total)), len(raw))


def test_total_smaller_than_the_question_count_gives_one_mark_each():
    marks, used = fit_marks(np.array([5.0, 5.0, 5.0, 5.0]), 2)
    assert list(marks) == [1, 1, 1, 1] and used == 4


@needs_topics
def test_damaged_and_duplicate_uploads_are_reported_not_raised():
    from src.inference.generator import PaperGenerator  # noqa: F401  (import check)
    from src.inference.library import Library
    from src.inference.predictor import Predictor
    from src.inference.uploads import ingest_papers
    subject = Predictor().subjects()[0]
    good_text = "\n".join(f"Q{i}. Explain the concept number {i} in detail and give two worked examples "
                          f"of it with a short justification of each step. (10 marks)" for i in range(1, 4))
    ok = {"name": "ok.txt", "data": good_text.encode(), "year": 2026, "assessment_type": "mid", "number": 1}
    for name, data in [("bad.png", b"not an image"), ("bad.jpg", b""), ("bad.docx", b"PK broken"),
                       ("bad.pdf", b"%PDF-1.4 broken"), ("empty.pdf", b""), ("x.exe", b"MZ")]:
        r = ingest_papers(subject, [ok, {**ok, "name": name, "data": data}])
        assert r.n_papers == 1 and any(name in w for w in r.warnings), name
    r = ingest_papers(subject, [ok, ok])
    assert r.n_papers == 1 and any("twice" in w for w in r.warnings)
    low = Predictor().low_confidence_subjects()[0]
    r = ingest_papers(low, [ok])
    assert r.n_papers == 0 and r.warnings
