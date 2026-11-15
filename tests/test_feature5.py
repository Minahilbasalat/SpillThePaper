"""Feature 5: read the answer sheet, mark it, find weak areas, build a personalised paper.
The language model is replaced by a fake, so no key or internet is needed."""
import json

import pytest

from src.inference import config as C
from src.inference.grading import (AnswerSheet, grade_paper, manual_grades, override, read_answers,
                                   split_answers)
from src.inference.personalised import generate_personalised, weak_focus
from src.inference.results import ResultsStore, analyse

PAPER = {"title": "T", "subject": "S", "exam_kind": "mid1", "source": "predicted", "questions": [
    {"number": 1, "text": "Explain TCP.", "marks": 10, "topic_id": 1, "topic_label": "tcp",
     "difficulty": "easy", "qtype": "theory", "solution": "SYN, SYN-ACK, ACK", "solution_source": "inline"},
    {"number": 2, "text": "Write quicksort.", "marks": 10, "topic_id": 2, "topic_label": "sorting",
     "difficulty": "hard", "qtype": "code", "solution": None, "solution_source": None},
    {"number": 3, "text": "Define a deadlock.", "marks": 5, "topic_id": 2, "topic_label": "sorting",
     "difficulty": "easy", "qtype": "theory", "solution": None, "solution_source": None}]}


class FakeLLM:
    available, supports_vision, model = True, True, "fake"

    def __init__(self, score=5, transcript=None, fail=False):
        self.score, self.transcript, self.fail, self.calls = score, transcript, fail, []

    def complete(self, system, user, max_tokens=1200):
        self.calls.append(user)
        if self.fail:
            raise RuntimeError("quota")
        return json.dumps({"score": self.score, "feedback": "ok", "missing": ["step 2"]})

    def complete_vision(self, system, user, images, max_tokens=2500):
        self.calls.append(("vision", len(images)))
        return json.dumps(self.transcript or {"answers": [], "notes": ""})


class NoAI:
    available, supports_vision, model = False, False, "none"


def test_split_answers():
    got, rest = split_answers("Name: Ali\nQ1. SYN then ACK\nmore\nQuestion 2: quick\nQ 3) deadlock", [1, 2, 3])
    assert got == {1: "SYN then ACK\nmore", 2: "quick", 3: "deadlock"} and rest == ""
    got, rest = split_answers("1. aaa\n2. bbb\n3. ccc", [1, 2, 3])
    assert set(got) == {1, 2, 3}
    got, rest = split_answers("just some text", [1, 2, 3])
    assert got == {} and rest == "just some text"
    assert split_answers("Q9. nope", [1, 2]) == ({}, "Q9. nope")        # unknown numbers are ignored
    assert split_answers("", [1]) == ({}, "")


def test_read_typed_answers_and_photos():
    s = read_answers(PAPER, [{"name": "a.txt", "data": b"Q1. three way\nQ3. circular wait"}], NoAI())
    assert s.answers == {1: "three way", 3: "circular wait"} and s.method == "text"
    assert any("question(s) 2" in w for w in s.warnings)
    llm = FakeLLM(transcript={"answers": [{"number": 2, "text": "def qs(a): ..."}, {"number": 9, "text": "x"},
                                          {"number": "bad"}], "notes": "page 2 blurry"})
    s = read_answers(PAPER, [{"name": "p1.jpg", "data": b"\xff\xd8x"}, {"name": "p2.png", "data": b"x"}], llm)
    assert s.answers == {2: "def qs(a): ..."} and s.method == "vision" and llm.calls == [("vision", 2)]
    assert any("blurry" in w for w in s.warnings)


def test_photos_without_vision_and_bad_files_are_warnings_not_errors():
    s = read_answers(PAPER, [{"name": "p.jpg", "data": b"not an image"}, {"name": "x.exe", "data": b"MZ"},
                             {"name": "b.pdf", "data": b"%PDF broken"}], NoAI())
    assert s.answers == {} and len(s.warnings) >= 3
    bad = FakeLLM(transcript=None)
    bad.complete_vision = lambda *a, **k: "I cannot read this"
    s = read_answers(PAPER, [{"name": "p.jpg", "data": b"x"}], bad)
    assert s.answers == {} and any("could not be read" in w for w in s.warnings)


def test_grading_with_ai():
    llm = FakeLLM(score=7.3)
    g = grade_paper(PAPER, {1: "SYN SYN-ACK ACK", 2: "[illegible]"}, llm=llm)
    assert [x["status"] for x in g] == ["graded_ai", "unreadable", "unanswered"]
    assert g[0]["score"] == 7.5 and g[0]["missing"] == ["step 2"] and g[0]["reference"] == "archive"
    assert g[1]["score"] is None and g[2]["score"] == 0.0
    assert "student's answer" in llm.calls[0].lower() and "SYN, SYN-ACK, ACK" in llm.calls[0]
    # a score above the maximum is clamped, a negative one too
    assert grade_paper(PAPER, {1: "x"}, llm=FakeLLM(score=99))[0]["score"] == 10.0
    assert grade_paper(PAPER, {1: "x"}, llm=FakeLLM(score=-3))[0]["score"] == 0.0


def test_grading_without_ai_or_when_ai_fails():
    g = grade_paper(PAPER, {1: "something"}, llm=NoAI())
    assert g[0]["status"] == "not_graded" and g[0]["score"] is None and g[2]["status"] == "unanswered"
    g = grade_paper(PAPER, {1: "something"}, llm=FakeLLM(fail=True))
    assert g[0]["status"] == "not_graded" and "failed" in g[0]["feedback"]
    g = grade_paper(PAPER, {1: "x"}, llm=FakeLLM(), allow_ai=False)
    assert g[0]["status"] == "not_graded"


def test_manual_marks_and_override():
    g = manual_grades(PAPER, {1: 6, 2: 100, 3: None})
    assert [x["status"] for x in g] == ["manual", "manual", "not_graded"] and g[1]["score"] == 10.0
    g2 = grade_paper(PAPER, {1: "x"}, llm=FakeLLM(score=2))
    g3 = override(g2, 1, 8)
    assert g3[0]["score"] == 8 and g3[0]["status"] == "manual" and "AI said 2" in g3[0]["feedback"]
    assert g2[0]["score"] == 2                                         # the original is untouched


def test_analysis_finds_weak_topics():
    g = manual_grades(PAPER, {1: 9, 2: 2, 3: 1})
    a = analyse(PAPER, g)
    assert a["overall"] == {"score": 12.0, "total": 25.0, "percent": 48.0, "questions": 3, "counted": 3}
    assert a["weak_topic_ids"] == [2] and a["topics"][0]["topic_id"] == 2 and a["topics"][0]["percent"] == 20.0
    assert a["topics"][1]["band"] == "strong" and a["topics"][1]["confidence"] == "low"
    assert a["by_difficulty"]["hard"]["percent"] == 20.0 and a["not_counted"] == []
    with pytest.raises(ValueError):
        analyse(PAPER, manual_grades(PAPER, {}))
    part = analyse(PAPER, manual_grades(PAPER, {1: 5}))                 # only one question marked
    assert part["not_counted"] == [2, 3] and part["overall"]["total"] == 10.0


def test_store_history_and_cumulative_weakness(tmp_path):
    st = ResultsStore(tmp_path / "r.json")
    assert st.attempts() == [] and st.weakness("S") == {}
    a1 = analyse(PAPER, manual_grades(PAPER, {1: 10, 2: 0, 3: 0}))
    i1 = st.add(PAPER, manual_grades(PAPER, {1: 10, 2: 0, 3: 0}), a1)
    i2 = st.add(PAPER, manual_grades(PAPER, {1: 0, 2: 10, 3: 5}))
    assert [r["id"] for r in st.attempts("S")] == [i2, i1] and st.attempts("other") == []
    w = st.weakness("S")
    assert w[1]["weakness"] < 1 and w[2]["weakness"] < 1 and w[1]["attempts"] == 2
    assert st.delete(i1) and not st.delete("nope") and len(st.attempts()) == 1
    (tmp_path / "r.json").write_text("{broken")                          # a damaged file never crashes
    assert st.attempts() == []


def test_weak_focus_mixes_current_and_history(tmp_path):
    st = ResultsStore(tmp_path / "r.json")
    st.add(PAPER, manual_grades(PAPER, {1: 10, 2: 0, 3: 0}))
    a = analyse(PAPER, manual_grades(PAPER, {1: 10, 2: 1, 3: 1}))
    f = weak_focus("S", a, st)
    assert list(f) == [2] and f[2]["weakness"] > 0.8
    assert weak_focus("S", analyse(PAPER, manual_grades(PAPER, {1: 10, 2: 9, 3: 5}))) == {}
    with pytest.raises(ValueError):
        generate_personalised(None, "S", "mid1", 2026)


needs_runtime = pytest.mark.skipif(not C.QUESTION_TABLE_CSV.exists() or not (C.MODELS_DIR / "topics").exists(),
                                   reason="runtime bundle missing")


@needs_runtime
def test_personalised_paper_on_real_data():
    from src.inference.generator import PaperGenerator
    from src.inference.library import Library
    from src.inference.predictor import Predictor
    gen = PaperGenerator(Predictor(), Library())
    subject = "Computer Networks" if "Computer Networks" in gen.pred.subjects() else gen.pred.subjects()[0]
    base = gen.generate(subject, "mid1", 2026, seed=3)
    assert gen.generate(subject, "mid1", 2026, seed=3)["questions"] == base["questions"]      # unchanged
    assert not any(q["focus"] for q in base["questions"])
    grades = manual_grades(base, {q["number"]: (0 if i < 2 else q["marks"]) for i, q in enumerate(base["questions"])})
    a = analyse(base, grades)
    p = generate_personalised(gen, subject, "mid1", 2026, analysis=a, source_paper=base, seed=3)
    assert p["personalised"] and p["title"].startswith("Personalised") and p["focus"]
    weak = set(a["weak_topic_ids"])
    f = [q for q in p["questions"] if q["focus"]]
    assert f and all(q["topic_id"] in weak for q in f) and len(f) < len(p["questions"]) or len(p["questions"]) < 3
    assert not ({q["text"] for q in p["questions"]} & {q["text"] for q in base["questions"]})
    assert sum(q["marks"] for q in p["questions"]) == p["total_marks"]
    assert all(q["marks"] >= 1 for q in p["questions"])
    assert p["realised"]["focus_questions"] == len(f) and any("Personalised" in n for n in p["notes"])
