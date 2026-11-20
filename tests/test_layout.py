"""Layout of questions: sub-parts on their own lines, code in indented blocks, form reset."""
import sys
from pathlib import Path

import pytest

from src.inference.formatting import format_question
from src.inference.layout import beautify_code, blocks_html, layout_blocks, restore_parts

APP = Path(__file__).resolve().parents[1] / "app"

DAA = ("Answer the following questions, and in each case provide a brief explanation for your answer. "
       "No marks shall be awarded without a correct explanation. i. Is 2^(n+1) = O(2^n)? "
       "ii. Let f(n) = n log n and g(n) = n^2. Is f(n) = o(g(n))? "
       "iii. The recurrence T(n) = 2T(n/2) + n has solution Theta(n log n).")

OOP = ("Consider the following code and tell the output. class A { public: A() { cout << \"A\"; } "
       "virtual void show() { cout << \"show A\"; } }; class B : public A { public: B() { cout << \"B\"; } "
       "void show() { cout << \"show B\"; } }; int main() { A *p = new B(); p->show(); delete p; return 0; } "
       "What is the output of the above program? Explain.")


def test_q1_parts_start_on_new_lines():
    lines = restore_parts(DAA).split("\n")
    assert lines[0].endswith("without a correct explanation.")
    assert [ln.split()[0] for ln in lines[1:]] == ["i.", "ii.", "iii."]


def test_format_question_puts_parts_on_new_lines():
    lines = format_question(DAA).split("\n")
    assert len(lines) == 4 and lines[1].startswith("i. ") and lines[3].startswith("iii. ")


@pytest.mark.parametrize("text", [
    "Explain i.e. what a deadlock is. Is it avoidable?",
    "Compare Plan A. Plan B. And pick one.",
    "Consider T(n) = 2T(n/2) + n; solve it.",
    "Find f(i) for i = 1."])
def test_prose_is_not_split(text):
    assert restore_parts(text) == text


def test_squashed_code_becomes_an_indented_block():
    blocks = layout_blocks(OOP)
    assert [k for k, _ in blocks] == ["text", "code", "text"]
    assert blocks[0][1] == "Consider the following code and tell the output."
    assert blocks[2][1].startswith("What is the output")
    code = blocks[1][1].split("\n")
    assert code[0] == "class A {" and code[1] == "public:" and code[2] == "    A() {"
    assert '        cout << "A";' in code and code[-1] == "}" and "    return 0;" in code


def test_code_on_separate_lines_is_reindented():
    t = ("What is wrong with this code?\n#include <iostream>\nusing namespace std;\nclass P {\nint x;\npublic:\n"
         "P(int a) { x = a; }\nvoid f() {\nfor (int i = 0; i < 3; i++) {\ncout << i;\n}\n}\n};\nFix it.")
    kinds = layout_blocks(t)
    assert [k for k, _ in kinds] == ["text", "code", "text"]
    code = kinds[1][1].split("\n")
    assert code[0] == "#include <iostream>" and "        x = a;" in code and "            cout << i;" in code


def test_initialiser_braces_and_for_loops_stay_on_one_line():
    code = beautify_code("int main() { int a[3] = {1, 2, 3}; for (int i = 0; i < 3; i++) { cout << a[i]; } }")
    assert "    int a[3] = {1, 2, 3};" in code.split("\n")
    assert "    for (int i = 0; i < 3; i++) {" in code.split("\n")


@pytest.mark.parametrize("text", [DAA, OOP, "Plain question without structure.", "", "  "])
def test_layout_is_idempotent_and_keeps_every_word(text):
    again = "\n".join(b for _, b in layout_blocks(text))
    assert beautify_code(again) == beautify_code(beautify_code(again))
    assert restore_parts(restore_parts(text)) == restore_parts(text)
    import re
    words = lambda s: re.findall(r"[A-Za-z0-9_]+", s)
    assert words(again) == words(restore_parts(text))


def test_html_has_no_line_breaks_and_is_escaped():
    out = blocks_html(OOP + " <b>x</b>")
    assert "\n" not in out and "<pre" in out and "&lt;&lt;" in out and "<b>x</b>" not in out
    assert 'class="ql qpart"' in blocks_html(DAA)


def test_pdf_export_keeps_code_and_parts():
    pytest.importorskip("fpdf")
    from src.inference.export import paper_to_pdf
    paper = {"title": "T", "duration_minutes": 60, "total_marks": 10, "source": "predicted",
             "questions": [{"number": 1, "text": DAA, "marks": 5}, {"number": 2, "text": OOP, "marks": 5}]}
    assert paper_to_pdf(paper).startswith(b"%PDF")


def test_back_home_clears_the_predict_form():
    from streamlit.testing.v1 import AppTest
    script = f"""
import sys
sys.path.insert(0, {str(APP)!r})
import streamlit as st
import state
st.session_state["up_topics"] = ["Hashing"]
st.session_state["up_syll_sig"] = "a.pdf:10"
st.session_state["up_syll_items"] = ["x"]
st.session_state["up_upload"] = {{"subject": "x"}}
st.session_state["up_screen"] = "paper"
st.session_state["other"] = 1
state.reset_predict_form()
st.session_state["left"] = sorted(k for k in st.session_state if k.startswith("up_"))
"""
    at = AppTest.from_string(script).run()
    assert not at.exception
    assert at.session_state["left"] == []
    assert at.session_state["other"] == 1
