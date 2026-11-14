import time

import pytest

from src.data.clean_questions import is_instruction, strip_furniture


@pytest.mark.parametrize("text", [
    "1. Your cell phones/smart watches should be turned off and placed upside down.",
    "1. Understanding the question paper is also part of the exam, so do not ask any "
    "clarification. Make suitable ASSUMPTIONS.",
    "3. You are immediately disqualified from the exam if: i. You are seen talking, "
    "whispering, borrowing or looking at someone's PC ii. A USB is found attached",
    "5. Any kind of cheat sheet/code if found in your PC will result in immediate "
    "disqualification from Final Exam and F as final grade.",
    "",
    "Page 3",
])
def test_rules_are_instructions(text):
    assert is_instruction(strip_furniture(text))


@pytest.mark.parametrize("text", [
    "Q8: To be solved on the question paper. Refer to the diagram below, all host ARP "
    "caches are empty. Show the frames exchanged when host A pings host B. (10 marks)",
    "Q1. Identify all the classes and make a class diagram. Marks will be deducted if "
    "relations are missing. (15)",
    "Explain how a cheat sheet attack works against a weak hash function?",
    "Write a C++ program that reads a list of numbers and prints the median.",
])
def test_real_questions_are_kept(text):
    assert not is_instruction(strip_furniture(text))


def test_repeated_short_text_without_task_is_boilerplate():
    assert is_instruction("Final submissions go in your own folder", repeats=5)
    assert not is_instruction("Explain the TCP handshake?", repeats=5)


def test_furniture_removed():
    t = ("Q9. Write a program to reverse a string. (2 marks) | National University of "
         "Computer and Emerging Sciences, Lahore Campus | Page 3 of 7")
    out = strip_furniture(t)
    assert "National University" not in out and "Page 3" not in out
    assert out.startswith("Q9. Write a program") and out.endswith("(2 marks)")


def test_no_catastrophic_backtracking_on_long_text():
    t = "Question 2: (Marks = 20) " + "word  " * 3000 + "end"
    t0 = time.time()
    strip_furniture(t)
    assert time.time() - t0 < 2
