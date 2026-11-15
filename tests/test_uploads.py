import io

import numpy as np
import pandas as pd
import pytest

from src.inference import config as C
from src.inference.documents import extract_text

needs_topics = pytest.mark.skipif(not (C.MODELS_DIR / "topics").exists(), reason="topic models missing")


def _pdf(text):
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), text, fontsize=11)
    return doc.tobytes()


def _docx(text):
    import docx
    d = docx.Document()
    for line in text.split("\n"):
        d.add_paragraph(line)
    b = io.BytesIO()
    d.save(b)
    return b.getvalue()


PAPER = ("Q1. Explain the three way handshake used by TCP to open a connection and why it needs "
         "three messages instead of two so that both sides agree on initial sequence numbers. (5)\n"
         "Q2. A router receives a packet destined for 192.168.4.77. Show how longest prefix match "
         "chooses the outgoing link from the forwarding table given below in this question. (10)\n"
         "Q3. Compute the CRC for the bit stream 101011111 with generator x4+x3+x2+1 and show every "
         "division step of the calculation clearly. (8)\n") * 1


def test_extract_formats():
    assert "three way handshake" in extract_text("a.txt", PAPER.encode())["text"]
    assert "three way handshake" in extract_text("a.docx", _docx(PAPER))["text"]
    assert extract_text("a.docx", _docx(PAPER))["method"] == "docx"
    with pytest.raises(ValueError):
        extract_text("a.exe", b"x")


def test_text_pdf_is_not_ocred():
    r = extract_text("a.pdf", _pdf(PAPER * 2))
    assert r["method"] == "pdf_text" and "longest prefix" in r["text"]


@needs_topics
def test_saved_topic_models_reproduce_stored_topics():
    """The persisted TF-IDF + KMeans must give the SAME topic ids as the data build."""
    from src.inference.topic_assign import assign_topics
    q = pd.read_csv(C.QUESTION_TABLE_CSV, dtype={"file_id": str})
    for s in q.loc[q["topic_id"].notna(), "subject"].unique()[:6]:
        g = q[(q["subject"] == s) & q["topic_id"].notna()]
        again = assign_topics(s, g["q_text"].tolist())
        assert np.array_equal(again, g["topic_id"].astype(int).to_numpy()), s


@needs_topics
def test_ingest_docx_paper():
    from src.inference.uploads import ingest_papers
    r = ingest_papers("Computer Networks", [dict(name="mid.docx", data=_docx(PAPER), year=2026,
                                                 assessment_type="mid", number=1)])
    assert r.n_papers == 1 and len(r.questions) >= 3
    assert r.papers.iloc[0]["exam_kind"] == "mid1"
    assert r.incidence["topic_id"].between(0, 10).all()


@needs_topics
def test_bad_file_gives_warning_not_crash():
    from src.inference.uploads import ingest_papers
    r = ingest_papers("Computer Networks", [dict(name="x.exe", data=b"1", year=2026, assessment_type="mid", number=None)])
    assert r.n_papers == 0 and r.warnings
