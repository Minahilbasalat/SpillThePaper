"""Minimal client for the Anthropic Messages API (uses `requests`, no extra package).

Configuration (environment variables, never put a key in the code or in Git):
    ANTHROPIC_API_KEY   your key
    LLM_MODEL           optional, default below (check the current model names in the Anthropic docs)
The rest of the app only needs `LLMClient.available` and `LLMClient.complete(...)`, so another
provider can be swapped in by writing a class with the same two members."""
from __future__ import annotations

import os
import time

import requests

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 90):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        if not self.available:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        body = {"model": self.model, "max_tokens": max_tokens, "system": system,
                "messages": [{"role": "user", "content": user}]}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        last = ""
        for attempt in range(3):
            r = requests.post(API_URL, json=body, headers=headers, timeout=self.timeout)
            if r.status_code == 200:
                blocks = r.json().get("content", [])
                return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            last = f"{r.status_code}: {r.text[:200]}"
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(2 * (attempt + 1))
                continue
            break
        raise LLMError(f"the model request failed ({last})")
