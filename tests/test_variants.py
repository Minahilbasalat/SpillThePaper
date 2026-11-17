"""The AI step on a paper (new questions / tidied questions), one batched call. The AI is a fake."""
import json
import re

import pytest

from src.inference.variants import rewrite_paper

ORIG = ("Write a C++ program that reads n integers into a dynamic array, finds the largest value and the "
        "smallest value, and prints both of them. Use pointers for all array access and free the memory.")
NEW = ("Design a C++ function that receives a pointer to a dynamically allocated array of student marks and "
       "returns the average mark. Show how the array is created in main and released afterwards.")
NEW2 = ("Implement a C++ class Stack backed by a dynamically allocated array with push, pop and peek, and "
        "explain what happens to the memory when the stack is destroyed.")
TIDY = ("Write a C++ program that reads n integers into a dynamic array,\nfinds the largest value and the "
        "smallest value,\nand prints both of them.\nUse pointers for all array access and free the memory.")


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
    """script: one dict {number: question text or None} per call; the last one repeats."""
    available, provider, model = True, "fake", "m"

    def __init__(self, *script, diagram=False, fail=False):
        self.script, self.diagram, self.fail, self.calls, self.prompts = list(script), diagram, fail, 0, []

    def complete(self, system, user, max_tokens=1200):
        self.calls += 1
        self.prompts.append(user)
        if self.fail:
            raise RuntimeError("quota")
        plan = self.script[min(self.calls, len(self.script)) - 1]
        nums = [int(n) for n in re.findall(r"\[#(\d+)\]", user)]
        return json.dumps({"questions": [{"number": n, "question": plan.get(n), "needs_diagram": self.diagram}
                                         for n in nums if n in plan]})


def test_one_call_writes_the_whole_paper(tmp_path):
    ai = Fake({1: NEW, 2: NEW2})
    out = rewrite_paper(paper(), ai, "new", cache_path=tmp_path / "c.json")
    assert ai.calls == 1
    q1, q2 = out["questions"]
    assert q1["text"] == NEW and q2["text"] == NEW2 and q1["raw_text"] == ORIG
    assert q1["ai_written"] and q1["ai_mode"] == "new" and q1["based_on"] == "Final 2023" and q1["marks"] == 10
    assert q1["solution"] is None and q1["solution_source"] is None and q1["needs_figure"] is False
    assert out["question_style"] == "new" and out["notes"][0].startswith("AI-written")
    assert not any(n.startswith("These are real past") for n in out["notes"])
    assert paper()["questions"][0]["text"] == ORIG                                  # the input is untouched


def test_tidy_keeps_the_real_question_and_its_solution(tmp_path):
    out = rewrite_paper(paper(), Fake({1: TIDY}), "tidy", cache_path=tmp_path / "c.json")
    q = out["questions"][0]
    assert q["text"] == TIDY and q["raw_text"] == ORIG and q["ai_mode"] == "tidy" and not q.get("ai_written")
    assert q["solution"] == "int main(){}" and out["question_style"] == "tidy"
    assert out["questions"][1]["text"].startswith("Write a C++")                    # no answer for Q2: kept


def test_only_the_failed_question_is_asked_again(tmp_path):
    ai = Fake({1: NEW, 2: ORIG}, {2: NEW2})                  # first reply: Q2 is a copy, second call fixes it
    out = rewrite_paper(paper(), ai, "new", cache_path=tmp_path / "c.json")
    assert ai.calls == 2 and "[#1]" not in ai.prompts[1] and "[#2]" in ai.prompts[1]
    assert [q["text"] for q in out["questions"]] == [NEW, NEW2]


@pytest.mark.parametrize("reply", [ORIG, "Too short.", None,
                                   "As an AI I cannot write that question for you because it is not allowed."])
def test_bad_answers_keep_the_real_tidied_question(tmp_path, reply):
    out = rewrite_paper(paper(), Fake({1: reply, 2: reply}), "new", cache_path=tmp_path / "c.json")
    assert out["questions"][0]["text"] == ORIG and out["question_style"] == "real"
    assert any("kept the real past question" in n for n in out["notes"])


def test_diagram_questions_are_not_rewritten(tmp_path):
    out = rewrite_paper(paper(), Fake({1: NEW, 2: NEW2}, diagram=True), "new", cache_path=tmp_path / "c.json")
    assert out["questions"][0]["text"] == ORIG
    out = rewrite_paper(paper(), Fake({1: TIDY, 2: TIDY}, diagram=True), "tidy", cache_path=tmp_path / "d.json")
    assert out["questions"][0]["needs_figure"] is True


def test_ai_down_is_one_failed_call_and_the_cache_saves_calls(tmp_path):
    bad = Fake(fail=True)
    out = rewrite_paper(paper(), bad, "new", cache_path=tmp_path / "c.json")
    assert bad.calls == 1 and out["questions"][0]["text"] == ORIG and "not available" in out["notes"][-1]
    ok = Fake({1: NEW, 2: NEW2})
    rewrite_paper(paper(), ok, "new", cache_path=tmp_path / "d.json")
    again = rewrite_paper(paper(), ok, "new", cache_path=tmp_path / "d.json")
    assert ok.calls == 1 and again["questions"][0]["text"] == NEW                    # second time: from the cache


def test_no_key_bad_mode_and_running_twice(tmp_path):
    class Off:
        available = False
    out = rewrite_paper(paper(), Off(), "new", cache_path=tmp_path / "c.json")
    assert out["questions"][0]["text"] == ORIG and any("switched off" in n for n in out["notes"])
    with pytest.raises(ValueError):
        rewrite_paper(paper(), Fake({}), "other")
    ok = Fake({1: NEW, 2: NEW2})
    once = rewrite_paper(paper(), ok, "new", cache_path=tmp_path / "e.json")
    assert rewrite_paper(once, ok, "new", cache_path=tmp_path / "e.json") is once and ok.calls == 1


def test_progress_callback(tmp_path):
    seen = []
    rewrite_paper(paper(), Fake({1: NEW, 2: NEW2}), "new", progress=lambda i, n: seen.append((i, n)),
                  cache_path=tmp_path / "c.json")
    assert seen == [(2, 2)]
