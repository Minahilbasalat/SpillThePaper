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


# A course description form (NCEAC style): no "Course Outline" heading, topics in a Week/Lecture Details table.
FORM = """National Computing Education Accreditation Council
NCEAC
COURSE DESCRIPTION FORM
INSTITUTION: FAST-NUCES
A. Course Description
Course Code
CS2009
Course Title
Design and Analysis of Algorithms
Prerequisites by Course(s)
and Topics
CS2001 Data Structures
Assessment Instruments
with Weights (homework,
quizzes, midterms, final,
1. Assignments (10 %)
2. Class Participation (5 %)
4. Midterms (30 %)
Course Instructor
Dr. Tahir Ejaz
Office Hours
Mondays & Thursdays 4:00-5:00 pm
Topics to be Covered in the
Course, (assume 15-week
instruction and one-and-a-
half-hour lectures)
Week
Lecture Details
Book
Chapters
1-5
\u2212
RAM Model
\u2212
Insertion Sort
\u2212
Merge sort
\u2212
Asymptotic functions and Notations, best-case
and worst-case time complexity
\u2212
Solving recurrences
1-4,7,8
6-12
Dynamic Programming
\u2212
Rod Cutting
\u2212
Longest Common Subsequence
NCEAC.FORM.001.D
National Computing Education Accreditation Council
NCEAC
13-16
Minimum Spanning Trees
\u2212
Prim's Algorithm
22-25
Laboratory
Projects/Experiments Done
in the Course
N/A
Instructor Name: Dr. Tahir Ejaz
Date: 21-Aug-2026
"""


def test_course_description_form_keeps_only_the_topics():
    items = parse_syllabus(FORM)["items"]
    assert items == ["RAM Model", "Insertion Sort", "Merge sort",
                     "Asymptotic functions and Notations, best-case and worst-case time complexity",
                     "Solving recurrences", "Dynamic Programming", "Rod Cutting",
                     "Longest Common Subsequence", "Minimum Spanning Trees", "Prim's Algorithm"]