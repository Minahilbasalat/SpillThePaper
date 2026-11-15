import io

import pytest

from src.inference import config as C
from src.inference.errors import LowConfidenceError

pytestmark = pytest.mark.skipif(
    not (C.QUESTION_TABLE_CSV.exists() and (C.MODELS_DIR / "preprocess.joblib").exists()
         and (C.MODELS_DIR / "topics").exists()), reason="runtime data not present")


@pytest.fixture(scope="module")
def gen():
    from src.inference.generator import PaperGenerator
    from src.inference.library import Library
    from src.inference.predictor import Predictor
    return PaperGenerator(Predictor(), Library())


@pytest.fixture(scope="module")
def subject(gen):
    return max(gen.pred.subjects(), key=lambda s: gen.lib.overview(s)["papers"])


@pytest.mark.parametrize("exam", ["quiz", "mid1", "mid2", "final"])
def test_paper_follows_blueprint(gen, subject, exam):
    p = gen.generate(subject, exam, 2026)
    bp = p["blueprint"]
    assert len(p["questions"]) == max(2, round(bp["questions"]))
    assert abs(sum(q["marks"] for q in p["questions"]) - p["total_marks"]) < 1e-6
    texts = [q["text"] for q in p["questions"]]
    assert len(set(texts)) == len(texts) and all(len(t) > 10 for t in texts)
    assert [q["number"] for q in p["questions"]] == list(range(1, len(texts) + 1))
    assert p["duration_minutes"] == {"quiz": 30, "mid1": 60, "mid2": 60, "final": 180}[exam]
    assert p["source"] == "predicted" and p["notes"]


def test_includes_less_predicted_areas_when_paper_is_big_enough(gen, subject):
    p = gen.generate(subject, "final", 2026, n_questions=8)
    assert p["realised"]["explore_questions"] >= 1


def test_reproducible_and_seed_changes_paper(gen, subject):
    a = gen.generate(subject, "final", 2026)
    b = gen.generate(subject, "final", 2026)
    c = gen.generate(subject, "final", 2026, seed=123)
    assert [q["text"] for q in a["questions"]] == [q["text"] for q in b["questions"]]
    assert [q["text"] for q in a["questions"]] != [q["text"] for q in c["questions"]]


def test_syllabus_is_used_and_reported(gen, subject):
    p = gen.generate(subject, "final", 2026, syllabus_items=["congestion control", "routing algorithms", "TCP"])
    assert all(q["syllabus_match"] is not None for q in p["questions"])
    assert any("Syllabus matching" in n for n in p["notes"])


def test_unsupported_subject_raises(gen):
    with pytest.raises(LowConfidenceError):
        gen.generate(gen.pred.low_confidence_subjects()[0], "mid1", 2026)


def test_uploaded_paper_changes_history_and_flags_examined_topics(gen):
    import docx
    from src.inference.uploads import ingest_papers
    d = docx.Document()
    for line in ["Q1. Explain how TCP congestion control uses slow start and congestion avoidance "
                 "to adapt the sending rate of a connection to the available capacity. (10)",
                 "Q2. Show how the Dijkstra shortest path algorithm computes the routing table "
                 "for the following network topology with the given link costs. (10)"]:
        d.add_paragraph(line)
    b = io.BytesIO(); d.save(b)
    up = ingest_papers("Computer Networks", [dict(name="m1.docx", data=b.getvalue(), year=2026,
                                                  assessment_type="mid", number=1)])
    p = gen.generate("Computer Networks", "final", 2026, extra=up)
    assert p["topics"]["examined_this_year"].any()
    assert p["topics"].attrs["uploaded_papers"] == 1


def test_question_fields_for_the_ui(gen, subject):
    p = gen.generate(subject, "final", 2026)
    for q in p["questions"]:
        assert 1 <= q["likelihood"] <= 5 and isinstance(q["leetcode_style"], bool)
        assert "solution_source" in q
    assert all(q["likelihood"] == 1 for q in p["questions"] if q["explore"])
