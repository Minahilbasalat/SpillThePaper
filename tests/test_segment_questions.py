from src.data.segment_questions import segment_questions


def test_question_markers():
    text = (
        "Name: ____\nRoll No: ____\n"
        "Question 1 [10 marks] Explain the OSI model layers in detail.\n"
        "Question 2 [10 marks] What is the difference between TCP and UDP?\n"
    )
    qs, method = segment_questions(text)
    assert method == "marker"
    assert len(qs) == 2
    assert qs[0].startswith("Question 1")
    assert "Roll" not in qs[0]


def test_numbered_items():
    text = "1. Define a stack and its operations.\n2. Write a queue implementation.\n"
    qs, method = segment_questions(text)
    assert method == "numbered"
    assert len(qs) == 2


def test_paragraph_fallback():
    text = "Explain deadlock and the four conditions for it to occur in an operating system today.\n\n" \
           "Describe how paging differs from segmentation with a neat labelled diagram of both."
    qs, method = segment_questions(text)
    assert method == "paragraph"
    assert len(qs) >= 1

def test_drops_solutions_and_grading_table():
    text = (
        "Q1: Explain the OSI model layers in detail.\n"
        "Q2: What is the difference between TCP and UDP?\n"
        "Q2 Solution: TCP is connection oriented while UDP is not.\n"
    )
    qs, _ = segment_questions(text)
    assert len(qs) == 2
    assert not any("Solution" in q for q in qs)    

def test_drops_header_chunks():
    text = (
        "National University of Computer and Emerging Sciences\n\n"
        "Course Code: CS3001 Semester: Fall 2022 Section: A Roll No: ____\n\n"
        "Explain the difference between TCP and UDP with one real example each.\n\n"
        "Describe how a router uses its forwarding table to move packets onward."
    )
    qs, _ = segment_questions(text)
    assert qs, "real questions must survive"
    assert not any("Roll No" in q or "National University" in q for q in qs)    