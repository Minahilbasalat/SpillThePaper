import json

import pytest

from src.inference import llm as llm_module
from src.inference.export import paper_to_pdf
from src.inference.llm import LLMClient, LLMError
from src.inference.solutions import SolutionService

Q = {"number": 1, "text": "Explain the three way handshake used by TCP.", "marks": 5,
     "topic_label": "tcp", "solution": None, "solution_source": None, "low_quality": False}


class FakeLLM:
    available = True
    model = "fake"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens=1200):
        self.calls += 1
        return json.dumps({"solution": "SYN, SYN-ACK, ACK.", "marking_scheme": "1 mark per message"})


def test_archive_beats_ai(tmp_path):
    llm = FakeLLM()
    svc = SolutionService(llm, tmp_path)
    r = svc.solve("CN", {**Q, "solution": "Answer: 42", "solution_source": "companion"})
    assert r["source"] == "archive" and "partial" in r["note"] and llm.calls == 0


def test_ai_solution_is_cached_and_labelled(tmp_path):
    llm = FakeLLM()
    svc = SolutionService(llm, tmp_path)
    a, b = svc.solve("CN", Q), svc.solve("CN", Q)
    assert a["source"] == "ai" and "verify" in a["note"] and a["marking_scheme"]
    assert llm.calls == 1 and a == b
    assert SolutionService(FakeLLM(), tmp_path).solve("CN", Q)["text"] == a["text"]   # survives restart


def test_without_ai_the_answer_is_honest(tmp_path):
    svc = SolutionService(LLMClient(api_key=""), tmp_path)
    r = svc.solve("CN", Q)
    assert r["source"] == "none" and "ANTHROPIC_API_KEY" in r["note"]


def test_garbled_question_is_not_solved(tmp_path):
    llm = FakeLLM()
    r = SolutionService(llm, tmp_path).solve("CN", {**Q, "low_quality": True})
    assert r["source"] == "none" and llm.calls == 0


def test_plain_text_reply_is_accepted(tmp_path):
    class Plain(FakeLLM):
        def complete(self, *a, **k):
            return "Just text, not json."
    r = SolutionService(Plain(), tmp_path).solve("CN", Q)
    assert r["text"] == "Just text, not json." and r["marking_scheme"] is None


def test_llm_client_calls_the_api_and_reports_errors(monkeypatch):
    class Resp:
        def __init__(self, code, body): self.status_code, self._b, self.text = code, body, str(body)
        def json(self): return self._b
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return Resp(200, {"content": [{"type": "text", "text": "hello"}]})
    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    c = LLMClient(api_key="k", model="m")
    assert c.complete("sys", "usr") == "hello"
    assert sent["headers"]["x-api-key"] == "k" and sent["body"]["model"] == "m"
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: Resp(400, {"error": "bad"}))
    with pytest.raises(LLMError):
        c.complete("sys", "usr")
    with pytest.raises(LLMError):
        LLMClient(api_key="").complete("s", "u")


def test_pdf_export_with_solutions():
    paper = {"title": "Computer Networks - Mid 1 2026", "duration_minutes": 60, "total_marks": 30.0,
             "source": "predicted", "questions": [{"number": 1, "text": "Explain TCP \u2013 handshake \u2713", "marks": 5.0}]}
    pdf = paper_to_pdf(paper, [{"number": 1, "text": "SYN, SYN-ACK, ACK", "source": "ai", "marking_scheme": "1 each"}])
    assert pdf[:4] == b"%PDF" and len(pdf) > 1500
