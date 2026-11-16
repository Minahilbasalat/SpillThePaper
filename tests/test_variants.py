"""The optional AI step (tidy / new questions). The AI is replaced by a fake: no key, no internet."""
import json

import pytest

from src.inference.variants import rewrite_paper

ORIG = ("Write a C++ program that reads n integers into a dynamic array, finds the largest value and the "
        "smallest value, and prints both of them. Use pointers for all array access and free the memory.")


def paper():
    return {"subject": "OOP", "notes": ["These are real past questions chosen for the predicted topics; they "
                                         "are not newly written questions."],
            "questions": [{"number": 1, "text": ORIG, "marks": 10, "qtype": "code", "difficulty": "medium",
                           "topic_label": "pointers", "solution": "int main(){}", "solution_source": "inline",
                           "source": "Final 2023", "needs_figure": True},
                          {"number": 2, "text": ORIG + " Also count the values.", "marks": 5, "qtype": "code",
                           "difficulty": "easy", "topic_label": "arrays", "solution": None, "solution_source": None,
                           "source": "Mid 1 2022"}]}


class Fake:
    available, provider, model = True, "fake", "m"

    def __init__(self, question=None, diagram=False, fail=False):
        self.question, self.diagram, self.fail, self.calls = question, diagram, fail, 0

    def complete(self, system, user, max_tokens=1200):
        self.calls += 1
        if self.fail:
            raise RuntimeError("quota")
        return json.dumps({"question": self.question, "needs_diagram": self.diagram})


TIDY = "Write a C++ program that reads n integers into a dynamic array,\nfinds the largest value and the smallest value,\nand prints both of them.\nUse pointers for all array access and free the memory."
NEW = ("Design a C++ function that receives a pointer to a dynamically allocated array of student marks and "
       "returns the average mark. Show how the array is created in main and released afterwards.")


def test_tidy_keeps_the_real_question(tmp_path):
    out = rewrite_paper(paper(), Fake(TIDY), "tidy", cache_path=tmp_path / "c.json")
    q = out["questions"][0]
    assert q["text"] == TIDY and q["raw_text"] == ORIG and q["ai_mode"] == "tidy" and not q.get("ai_written")
    assert q["solution"] == "int main(){}" and q["needs_figure"] is False         # same question: solution stays
    assert out["question_style"] == "tidy" and any("tidied by the AI" in n for n in out["notes"])
    assert paper()["questions"][0]["text"] == ORIG                                   # input untouched


def test_tidy_that_changes_the_question_is_rejected(tmp_path):
    out = rewrite_paper(paper(), Fake(NEW), "tidy", cache_path=tmp_path / "c.json")
    assert out["questions"][0]["text"] == ORIG and out["question_style"] == "real"
    assert any("could not be tidied" in n for n in out["notes"])


def test_new_question_replaces_text_and_drops_the_old_solution(tmp_path):
    out = rewrite_paper(paper(), Fake(NEW), "new", cache_path=tmp_path / "c.json")
    q = out["questions"][0]
    assert q["text"] == NEW and q["raw_text"] == ORIG and q["ai_written"] and q["based_on"] == "Final 2023"
    assert q["solution"] is None and q["solution_source"] is None and q["marks"] == 10
    assert out["question_style"] == "new" and out["notes"][0].startswith("AI-written")
    assert not any(n.startswith("These are real past") for n in out["notes"])


@pytest.mark.parametrize("reply, why", [(ORIG, "a copy"), ("Too short.", "short"), ("As an AI I cannot write that "
                                        "question for you because it is not allowed in this context.", "refusal"),
                                        (None, "no question")])
def test_bad_new_questions_keep_the_real_one(tmp_path, reply, why):
    out = rewrite_paper(paper(), Fake(reply), "new", cache_path=tmp_path / "c.json")
    assert [q["text"] for q in out["questions"]][0] == ORIG, why
    assert out["question_style"] == "real" and any("could not be rewritten" in n for n in out["notes"])


def test_diagram_questions_are_not_rewritten(tmp_path):
    out = rewrite_paper(paper(), Fake(NEW, diagram=True), "new", cache_path=tmp_path / "c.json")
    assert out["questions"][0]["text"] == ORIG
    out = rewrite_paper(paper(), Fake(TIDY, diagram=True), "tidy", cache_path=tmp_path / "c.json")
    assert out["questions"][0]["needs_figure"] is True


def test_ai_down_stops_after_one_failed_call_and_cache_saves_calls(tmp_path):
    p = paper()
    p["questions"] += [dict(p["questions"][0], number=n) for n in (3, 4, 5)]
    bad = Fake(fail=True)
    out = rewrite_paper(p, bad, "new", cache_path=tmp_path / "c.json")
    assert bad.calls == 1 and all(q["text"] in (ORIG, ORIG + " Also count the values.") for q in out["questions"])
    ok = Fake(NEW)
    rewrite_paper(paper(), ok, "new", cache_path=tmp_path / "d.json")
    n = ok.calls
    again = rewrite_paper(paper(), ok, "new", cache_path=tmp_path / "d.json")
    assert ok.calls == n and again["questions"][0]["text"] == NEW                       # served from the cache


def test_no_key_and_bad_mode(tmp_path):
    class Off:
        available = False
    out = rewrite_paper(paper(), Off(), "new", cache_path=tmp_path / "c.json")
    assert out["questions"][0]["text"] == ORIG and any("switched off" in n for n in out["notes"])
    with pytest.raises(ValueError):
        rewrite_paper(paper(), Fake(NEW), "other")


def test_running_twice_does_not_repeat_the_work(tmp_path):
    ok = Fake(NEW)
    once = rewrite_paper(paper(), ok, "new", cache_path=tmp_path / "c.json")
    n = ok.calls
    twice = rewrite_paper(once, ok, "new", cache_path=tmp_path / "c.json")
    assert ok.calls == n and twice is once


def test_progress_callback(tmp_path):
    seen = []
    rewrite_paper(paper(), Fake(NEW), "new", progress=lambda i, n: seen.append((i, n)), cache_path=tmp_path / "c.json")
    assert seen == [(1, 2), (2, 2)]
