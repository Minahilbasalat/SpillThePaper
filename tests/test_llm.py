"""The language-model client: provider choice, request format, retries and readable errors.
No network and no key are needed: requests.post is replaced by a fake."""
import json

import pytest
import requests

from src.inference import llm as L
from src.inference.llm import LLMClient, LLMError, pick_provider
from src.inference.solutions import SolutionService

KEYS = ("GEMINI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in KEYS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)


class Resp:
    def __init__(self, code=200, body=None, headers=None, text=None):
        self.status_code, self._body, self.headers = code, body, headers or {}
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def openai_reply(text):
    return Resp(200, {"choices": [{"message": {"content": text}}]})


def test_provider_choice(monkeypatch):
    assert pick_provider({}) is None and LLMClient().available is False
    assert pick_provider({"GROQ_API_KEY": "x", "MISTRAL_API_KEY": "y"}) == "groq"        # first set key wins
    assert pick_provider({"GEMINI_API_KEY": "x", "GROQ_API_KEY": "y"}) == "gemini"
    assert pick_provider({"GEMINI_API_KEY": "x", "LLM_PROVIDER": "mistral"}) == "mistral"
    with pytest.raises(LLMError, match="LLM_PROVIDER"):
        pick_provider({"LLM_PROVIDER": "chatgpt"})
    monkeypatch.setenv("GROQ_API_KEY", "k")
    c = LLMClient()
    assert c.available and c.provider == "groq" and c.model == "llama-3.3-70b-versatile"
    monkeypatch.setenv("LLM_MODEL", "my-model")
    assert LLMClient().model == "my-model"
    monkeypatch.delenv("GROQ_API_KEY")
    assert LLMClient(api_key="k").provider == "anthropic"                                # old behaviour


@pytest.mark.parametrize("provider, host", [("gemini", "generativelanguage.googleapis.com"),
                                            ("groq", "api.groq.com"), ("mistral", "api.mistral.ai")])
def test_openai_style_providers(monkeypatch, provider, host):
    sent = {}

    def post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return openai_reply("  the answer  ")
    monkeypatch.setattr(L.requests, "post", post)
    c = LLMClient(provider=provider, api_key="secret")
    assert c.complete("be brief", "question?", 1000) == "the answer"
    assert host in sent["url"] and sent["headers"]["Authorization"] == "Bearer secret"
    assert sent["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["body"]["messages"][1]["content"] == "question?" and sent["body"]["model"] == c.model
    assert sent["body"]["max_tokens"] == (3000 if provider == "gemini" else 1000)


def test_retry_after_then_success(monkeypatch):
    replies = iter([Resp(429, {"e": 1}, {"retry-after": "3"}), Resp(503, {"e": 1}), openai_reply("ok")])
    monkeypatch.setattr(L.requests, "post", lambda *a, **k: next(replies))
    assert LLMClient(provider="groq", api_key="k").complete("s", "u") == "ok"


@pytest.mark.parametrize("reply, hint", [
    (Resp(401, {"e": 1}), "check the API key"),
    (Resp(404, {"e": 1}), "LLM_MODEL"),
    (Resp(429, {"e": 1}), "rate limit"),
])
def test_http_errors_have_readable_messages(monkeypatch, reply, hint):
    monkeypatch.setattr(L.requests, "post", lambda *a, **k: reply)
    with pytest.raises(LLMError, match=hint):
        LLMClient(provider="gemini", api_key="k").complete("s", "u")


def test_network_failures_become_llm_errors(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("no internet")
    monkeypatch.setattr(L.requests, "post", boom)
    with pytest.raises(LLMError, match="ConnectionError"):
        LLMClient(provider="groq", api_key="k").complete("s", "u")
    assert len(calls) == 3
    monkeypatch.setattr(L.requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout()))
    with pytest.raises(LLMError):
        LLMClient(provider="groq", api_key="k").complete("s", "u")


@pytest.mark.parametrize("reply", [Resp(200, {"choices": []}), Resp(200, {"choices": [{"message": {"content": None}}]}),
                                   Resp(200, None, text="<html>"), Resp(200, {"unexpected": True}),
                                   openai_reply("   ")])
def test_empty_or_malformed_answers(monkeypatch, reply):
    monkeypatch.setattr(L.requests, "post", lambda *a, **k: reply)
    with pytest.raises(LLMError, match="empty answer"):
        LLMClient(provider="mistral", api_key="k").complete("s", "u")


def test_a_failing_llm_never_crashes_the_solution_page(monkeypatch, tmp_path):
    q = {"number": 1, "text": "Explain TCP congestion control in detail.", "marks": 5, "solution": None,
         "low_quality": False, "topic_label": "tcp"}
    monkeypatch.setattr(L.requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    r = SolutionService(LLMClient(provider="groq", api_key="k"), tmp_path).solve("CN", q)
    assert r["source"] == "none" and "failed" in r["note"]
    monkeypatch.setattr(L.requests, "post", lambda *a, **k: openai_reply(""))   # empty: say so, do not cache
    svc = SolutionService(LLMClient(provider="groq", api_key="k"), tmp_path)
    assert svc.solve("CN", q)["source"] == "none"
    monkeypatch.setattr(L.requests, "post", lambda *a, **k: openai_reply("SYN, SYN-ACK, ACK"))
    assert svc.solve("CN", q)["source"] == "ai"


def test_vision_request_formats(monkeypatch):
    sent = {}

    def post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return openai_reply('{"answers": []}') if "anthropic" not in url else Resp(
            200, {"content": [{"type": "text", "text": "seen"}]})
    monkeypatch.setattr(L.requests, "post", post)
    g = LLMClient(provider="gemini", api_key="k")
    assert g.supports_vision and g.vision_model == g.model
    assert g.complete_vision("sys", "read this", [("image/jpeg", b"abc")]) == '{"answers": []}'
    parts = sent["body"]["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "read this"}
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,YWJj" and sent["body"]["temperature"] == 0.0
    a = LLMClient(provider="anthropic", api_key="k")
    assert a.complete_vision("sys", "read", [("image/png", b"abc")]) == "seen"
    blocks = sent["body"]["messages"][0]["content"]
    assert blocks[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "YWJj"}
    with pytest.raises(LLMError, match="no image"):
        g.complete_vision("s", "u", [])
    with pytest.raises(LLMError, match="no API key"):
        LLMClient().complete_vision("s", "u", [("image/png", b"x")])


def test_vision_model_choice(monkeypatch):
    assert LLMClient(provider="groq", api_key="k").vision_model.startswith("meta-llama")
    assert LLMClient(provider="mistral", api_key="k").vision_model == "pixtral-12b-2409"
    monkeypatch.setenv("LLM_VISION_MODEL", "my-vision")
    assert LLMClient(provider="groq", api_key="k").vision_model == "my-vision"
    assert not LLMClient().supports_vision
