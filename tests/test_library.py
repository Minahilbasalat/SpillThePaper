"""Library tests on the real data. Skipped where the derived data is missing."""
import pytest

from src.inference import config as C

pytestmark = pytest.mark.skipif(not C.QUESTION_TABLE_CSV.exists(), reason="runtime data not present")


@pytest.fixture(scope="module")
def lib():
    from src.inference.library import Library
    return Library()


def test_browse_all_subjects(lib):
    s = lib.all_subjects()
    assert len(s) > len(lib.supported()) and s["papers"].min() >= 1
    assert {"subject", "papers", "readable", "supported"} <= set(s.columns)


def test_exam_kinds_and_papers_have_real_links(lib):
    subj = lib.all_subjects().sort_values("papers").iloc[-1]["subject"]
    kinds = lib.exam_kinds(subj)
    assert kinds and {"kind", "label", "papers"} <= set(kinds[0])
    ps = lib.papers(subj, kinds[0]["kind"])
    assert len(ps) == kinds[0]["papers"]
    assert ps["original_url"].str.startswith("https://github.com/saleha-muzammil/").all()
    assert ps["original_url"].str.contains(C.ARCHIVE_COMMIT).all()


def test_overview_works_for_unsupported_subject(lib):
    s = lib.all_subjects()
    bad = s[~s["supported"]].iloc[0]["subject"]
    o = lib.overview(bad)
    assert not o["supported"] and o["reason_if_unsupported"]
    with pytest.raises(KeyError):
        lib.overview("No Such Subject")


def test_questions_are_clean(lib):
    s = max(lib.supported(), key=lambda x: lib.overview(x)["questions"])
    df = lib.questions(s, n=50)
    assert len(df) > 0 and not df["qtype"].isin(["fragment"]).any()


@pytest.mark.parametrize("exam,minutes", [("quiz", 30), ("mid1", 60), ("mid2", 60), ("final", 180)])
def test_blueprint(lib, exam, minutes):
    s = max(lib.supported(), key=lambda x: lib.overview(x)["papers"])
    b = lib.blueprint(s, exam)
    assert b["questions"] >= 1 and b["duration_minutes"] == minutes
    assert abs(sum(b["difficulty_mix"].values()) - 1) < 0.05


def test_past_paper_structure(lib):
    s = max(lib.supported(), key=lambda x: lib.overview(x)["papers"])
    ps = lib.papers(s)
    fid = ps[ps["questions"] > 0].iloc[0]["file_id"]
    p = lib.paper(fid, duration_minutes=45)
    assert p["duration_minutes"] == 45 and p["source"] == "past_paper"
    assert [q["number"] for q in p["questions"]] == list(range(1, len(p["questions"]) + 1))
    unreadable = ps[ps["questions"] == 0]
    if len(unreadable):
        with pytest.raises(KeyError):
            lib.paper(unreadable.iloc[0]["file_id"])


def test_new_fields_for_the_ui(lib):
    s = lib.all_subjects()
    assert "kinds" in s.columns and s["kinds"].str.len().max() > 0
    subj = s.sort_values("papers").iloc[-1]["subject"]
    ps = lib.papers(subj)
    assert ps["download_url"].str.startswith("https://raw.githubusercontent.com/").all()
    assert "total_marks" in ps.columns
    fid = ps[ps["questions"] > 0].iloc[0]["file_id"]
    q = lib.paper(fid)["questions"][0]
    assert {"solution_source", "low_quality"} <= set(q)
