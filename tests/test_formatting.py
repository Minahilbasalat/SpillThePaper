"""Readable layout for questions that were read from PDFs, and the "needs a diagram" flag."""
import re

import pytest

from src.inference import config as C
from src.inference.formatting import format_question, needs_figure

OOP = ("(CLO: 3) (Marks: 10) You are designing a library system. Your task is to create a dynamic 2D array "
       "according to the following conditions: ? Each row represents a section. ? Each column represents a genre. "
       "Your program needs to perform the following tasks: ? Identify the sections. ? Allocate memory. "
       "Write a C++ program. Example Input: int a[4][3]; // Function to create the array "
       "int** create(int a[][3]); // Function to free memory void free(int** p); a. Write the code. b. Explain it.")


def words(t):
    return re.findall(r"[A-Za-z0-9]+", t)


def test_tags_bullets_and_subparts():
    f = format_question(OOP)
    assert "CLO" not in f and "Marks:" not in f
    assert f.count("\n• ") == 4 and "?" not in f.replace("?\n", "")
    assert "\nExample Input:" in f and "\n// Function to create" in f and "\n// Function to free" in f
    assert "\na. Write the code." in f and "\nb. Explain it." in f
    assert words(f) == words(OOP)[words(OOP).index("You"):]            # nothing but the tags was dropped


@pytest.mark.parametrize("text", [OOP, "Question 3: (CLO: 3) (Marks: 10) Explain TCP. [5 marks]",
                                  "Q1) [CLO 2] What is a deadlock ? Explain it. (a) First (b) Second",
                                  "Plain question without any structure at all.", "", "   "])
def test_idempotent(text):
    f = format_question(text)
    assert format_question(f) == f


def test_real_question_marks_and_abbreviations_are_kept():
    t = "What is a deadlock? Explain it, e.g. with two processes. Is it avoidable? Why?"
    assert format_question(t) == t


def test_labels_and_end_tags_removed():
    assert format_question("Question 3: (CLO: 3) (Marks: 10) Explain TCP in detail please [5 marks]") \
        == "Explain TCP in detail please"
    assert format_question("Q4. Explain UDP") == "Explain UDP"


@pytest.mark.parametrize("text, expected", [
    ("Consider the circuit shown below and find the output.", True),
    ("Fig. 2 shows a class model. Identify the errors.", True),
    ("Look at the following diagram and name each part of it.", True),
    ("Give a value of A such that B is pruned. Is the sub-tree containing C and D pruned in the tree?", True),
    ("Which are challenges in the network? A. Remote operation. B. High power. C. Cost. D. Size.", False),
    ("Figure out the time complexity of binary search.", False),
    ("Draw a UML diagram for a library system.", False),
    ("A function takes an array and returns its sum. Write it in C++.", False),
])
def test_needs_figure(text, expected):
    assert needs_figure(text) is expected


needs_data = pytest.mark.skipif(not C.QUESTION_TABLE_CSV.exists(), reason="runtime bundle missing")


@needs_data
def test_every_real_question_formats_safely():
    import pandas as pd
    t = pd.read_csv(C.QUESTION_TABLE_CSV)["question_text"].astype(str)
    f = t.map(format_question)
    assert (f.str.len() > 0).all()
    assert (f.map(format_question) == f).all()
    assert 0.0 < t.map(needs_figure).mean() < 0.2                          # a minority, not everything
