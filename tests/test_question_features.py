import numpy as np
import pytest

from src.data.question_features import (bloom_level, parse_marks, question_type,
                                        split_solution)


@pytest.mark.parametrize("text,marks", [
    ("Explain TCP congestion control. (5 marks)", 5),
    ("Question 2 (3+7 marks) Consider the following ...", 10),
    ("Question 4: [5+2=7 Marks] a) Find the output", 7),
    ("Q1 [10] Define a deadlock and list the four necessary conditions.", 10),
    ("Question 1 (Marks = 10) GolfLabs is a dynamic company", 10),
    ("a) 4 marks Explain paging. b) 6 marks Explain segmentation.", 10),
    ("Why is the Ethernet minimum frame size required? Give two reasons (4)", 4),
])
def test_marks_parsed(text, marks):
    assert parse_marks(text)[0] == marks


def test_marks_missing_and_implausible():
    assert np.isnan(parse_marks("What is a process?")[0])
    assert np.isnan(parse_marks("Answer in 500 marks")[0])          # over the cap


def test_solution_split():
    q, s = split_solution("Q3) What is the largest number stored in 4 bytes using unary "
                          "encoding? Solution 4 bytes = 32 bits Largest number is 31")
    assert q.endswith("encoding?") and s.startswith("Solution")
    q, s = split_solution("Write a function to reverse a string. Answer: def rev(s): return s[::-1]")
    assert s.startswith("Answer:")


def test_no_solution_marker_in_plain_question():
    q, s = split_solution("Propose a solution to the dining philosophers problem and explain it.")
    assert s == ""


@pytest.mark.parametrize("text,expected", [
    ("1. Which of the following is true? a) one b) two c) three d) four", "mcq"),
    ("Write a C++ program to reverse a linked list using pointers.", "code"),
    ("Calculate the checksum of the following 16-bit words: 1011 0110 and 0101 1010.", "numerical"),
    ("Draw the ER diagram for a library management system.", "diagram_design"),
    ("Prove that the halting problem is undecidable.", "proof_derivation"),
    ("Explain the difference between TCP and UDP.", "theory"),
    ("State whether the following is True or False: every DFA is an NFA.", "true_false"),
    ("Arbitrary inputs of components are called __________ in React and Vue", "fill_blank"),
    ("A 3 4", "fragment"),
])
def test_question_type(text, expected):
    assert question_type(text) == expected


def test_bloom_levels():
    assert bloom_level("Define a semaphore.") == 1
    assert bloom_level("Explain how paging works.") == 2
    assert bloom_level("Design a schema for an online store.") == 6