"""Library tests on the real data. Skipped where the derived data is missing."""
import pytest

from src.inference import config as C

pytestmark = pytest.mark.skipif(not C.QUESTION_TABLE_CSV.exists(),
                                reason="runtime data not present")


@pytest.fixture(scope="module")
def lib():
    from src.inference.library import Library
    return Library()


def test_subjects_and_overview(lib):
    s = lib.subjects()[0]
    o = lib.overview(s)
    assert o["supported"] and o["papers"] > 0 and o["first_year"] <= o["last_year"]
    with pytest.raises(KeyError):
        lib.overview("No Such Subject")


def test_questions_are_clean(lib):
    s = max(lib.subjects(), key=lambda x: lib.overview(x)["questions"])
    df = lib.questions(s, n=50)
    assert len(df) > 0 and not df["qtype"].isin(["fragment"]).any()
    assert df["question_text"].str.len().min() > 0


def test_blueprint(lib):
    s = max(lib.subjects(), key=lambda x: lib.overview(x)["papers"])
    for t in ("quiz", "mid", "final"):
        b = lib.blueprint(s, t)
        assert b["questions"] >= 1 and b["duration_minutes"] == {"quiz": 30, "mid": 60, "final": 180}[t]
        assert abs(sum(b["question_type_shares"].values()) - 1) < 0.2


def test_past_paper_structure(lib):
    s = max(lib.subjects(), key=lambda x: lib.overview(x)["papers"])
    pp = lib.past_papers(s)
    assert len(pp) > 0
    p = lib.paper(pp.iloc[0]["file_id"], duration_minutes=45)
    assert p["duration_minutes"] == 45 and p["source"] == "past_paper"
    assert [q["number"] for q in p["questions"]] == list(range(1, len(p["questions"]) + 1))
    assert {"text", "marks", "qtype", "difficulty", "topic_label", "solution"} <= set(p["questions"][0])
