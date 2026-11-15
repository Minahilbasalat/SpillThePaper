from src.inference.syllabus import SyllabusMatcher, default_scope, parse_syllabus

OUTLINE = """Course Information
Credit hours: 3
Tentative Weekly Schedule
Week
Topics
1
Introduction and Revision of Basic Concepts
2
Finite Automata: DFA
3
Finite Automata: NFA
4
Regular Languages: Regular Expressions, Equivalence with FA
5
Regular Languages: Pumping Lemma
6
MIDTERM EXAM 1
7
Context-Free Languages: CFGs
8
Context-Free Languages: PDA
9
Context-Free Languages: Pumping Lemma
10
Context-Free Languages: DCFLs
11
MIDTERM EXAM 2
12
Turing Machines: Introduction
13
Decidability
(Tentative) Grading
1. Quizzes (20%)
2. Final Exam (40%)
"""


def test_weekly_schedule_and_midterm_boundaries():
    p = parse_syllabus(OUTLINE)
    assert p["items"][0].startswith("Introduction") and "Decidability" in p["items"][-1]
    assert not any("Credit" in i or "Grading" in i or "MIDTERM" in i for i in p["items"])
    assert p["mid1_boundary"] == 5 and p["mid2_boundary"] == 9
    assert default_scope(p, "mid1") == list(range(0, 5))
    assert default_scope(p, "mid2") == list(range(5, 9))
    assert default_scope(p, "final") == list(range(len(p["items"])))


def test_pasted_list_keeps_one_word_items():
    assert parse_syllabus("TCP, UDP, congestion control, routing algorithms, HTTP")["items"] == [
        "TCP", "UDP", "congestion control", "routing algorithms", "HTTP"]


def test_matcher_prefers_the_right_question():
    qs = ["Explain how TCP congestion control reacts to packet loss using slow start.",
          "Write an SQL query to join the customer and orders tables and group by city.",
          "Normalize the following relation up to third normal form."]
    s = SyllabusMatcher(qs, ["TCP congestion control"]).scores(qs)
    assert s[0] > s[1] and s[0] > s[2]
