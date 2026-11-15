"""Language-model client for AI-written solutions. Plain HTTPS through `requests`: nothing extra
to install, and several providers work (Gemini, Groq, Mistral, Anthropic).

Configuration = environment variables. NEVER put a key in the code, in a file that is committed,
or in a chat message.

    GEMINI_API_KEY   or  GROQ_API_KEY  or  MISTRAL_API_KEY  or  ANTHROPIC_API_KEY
    LLM_PROVIDER     optional: gemini | groq | mistral | anthropic
                     (default: the first of these whose key is set, in this order)
    LLM_VISION_MODEL optional: model that reads photos (default: Gemini/Anthropic read photos with
                     the normal model; Groq/Mistral use the model named in VISION_MODELS below)
    LLM_MODEL        optional: model name of that provider (see the defaults below; model names
                     change, if a request says "model not found" copy a current name from the
                     provider's dashboard)

Windows PowerShell (this window only):   $env:GEMINI_API_KEY = "paste-key-here"
Check that it works:                     python -m src.inference.llm

The rest of the app only needs `.available`, `.model` and `.complete(system, user, max_tokens)`.
Every failure (no internet, timeout, quota, bad key, empty answer) is raised as LLMError with a
readable message, so a page can show it instead of crashing."""
from __future__ import annotations

import base64
import os
import time

import requests

try:                                     # optional: read keys from a .env file in the project folder
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# provider -> (API key variable, URL, default model, wire format)
PROVIDERS = {
    "gemini": ("GEMINI_API_KEY",
               "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "gemini-flash-latest", "openai"),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions",
             "llama-3.3-70b-versatile", "openai"),
    "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1/chat/completions",
                "mistral-small-latest", "openai"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/messages",
                  "claude-sonnet-5-5", "anthropic"),
}
# Model used to READ IMAGES (handwritten answers). None = the normal model also reads images.
# Names change: if a request says "model not found", set LLM_VISION_MODEL to a current vision model.
VISION_MODELS = {"gemini": None, "anthropic": None,
                 "groq": "meta-llama/llama-4-scout-17b-16e-instruct", "mistral": "pixtral-12b-2409"}
RETRY_STATUS = (429, 500, 502, 503, 504, 529)
MAX_WAIT_S = 20


class LLMError(RuntimeError):
    pass


def pick_provider(env=None) -> str | None:
    env = os.environ if env is None else env
    wanted = (env.get("LLM_PROVIDER") or "").strip().lower()
    if wanted:
        if wanted not in PROVIDERS:
            raise LLMError(f"LLM_PROVIDER={wanted!r} is not one of {', '.join(PROVIDERS)}")
        return wanted
    return next((p for p, spec in PROVIDERS.items() if env.get(spec[0])), None)


class LLMClient:
    def __init__(self, provider: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: int = 90):
        self.provider = provider or pick_provider() or ("anthropic" if api_key else None)
        self.api_key = api_key or (os.environ.get(PROVIDERS[self.provider][0], "")
                                   if self.provider else "")
        default_model = PROVIDERS[self.provider][2] if self.provider else ""
        self.model = model or os.environ.get("LLM_MODEL") or default_model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.provider and self.api_key)

    def describe(self) -> str:
        return f"{self.provider} / {self.model}" if self.available else "no API key set"

    # -------------------------------------------------------------------- request
    def _request(self, system: str, user: str, max_tokens: int):
        key_var, url, _default, wire = PROVIDERS[self.provider]
        if wire == "anthropic":
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
            body = {"model": self.model, "max_tokens": max_tokens, "system": system,
                    "messages": [{"role": "user", "content": user}]}
        else:
            headers = {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
            # Gemini's newer models spend part of the budget on hidden "thinking": leave room
            budget = max_tokens * 3 if self.provider == "gemini" else max_tokens
            body = {"model": self.model, "max_tokens": budget, "temperature": 0.2,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}]}
        return url, headers, body, wire

    @staticmethod
    def _text(data: dict, wire: str) -> str:
        try:
            if wire == "anthropic":
                return "".join(b.get("text", "") for b in data.get("content", [])
                               if b.get("type") == "text").strip()
            return (data["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""

    def _send(self, url: str, headers: dict, body: dict, wire: str) -> str:
        """POST with retries; returns the answer text or raises LLMError with a readable message."""
        last = ""
        for attempt in range(3):
            try:
                r = requests.post(url, json=body, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:          # no internet, timeout, DNS ...
                last = f"{type(e).__name__}"
                time.sleep(min(2 * (attempt + 1), MAX_WAIT_S))
                continue
            if r.status_code == 200:
                try:
                    text = self._text(r.json(), wire)
                except ValueError:
                    text = ""
                if not text:
                    raise LLMError(f"{self.provider} returned an empty answer (try again, or set "
                                   "LLM_MODEL to another model)")
                return text
            last = f"{r.status_code}: {r.text[:200]}"
            if r.status_code in RETRY_STATUS and attempt < 2:
                try:
                    wait = float(r.headers.get("retry-after", 2 * (attempt + 1)))
                except ValueError:
                    wait = 2 * (attempt + 1)
                time.sleep(min(wait, MAX_WAIT_S))
                continue
            break
        hint = ""
        if last.startswith(("401", "403")):
            hint = " - check the API key"
        elif last.startswith("404"):
            hint = " - model not found: set LLM_MODEL (or LLM_VISION_MODEL) to a current model name"
        elif last.startswith("429"):
            hint = " - quota or rate limit reached: wait a minute and try again"
        raise LLMError(f"the {self.provider} request failed ({last}){hint}")

    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        if not self.available:
            raise LLMError("no API key is set (GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY or "
                           "ANTHROPIC_API_KEY)")
        url, headers, body, wire = self._request(system, user, max_tokens)
        return self._send(url, headers, body, wire)

    # ------------------------------------------------------------------- photos
    @property
    def vision_model(self) -> str:
        env = os.environ.get("LLM_VISION_MODEL")
        if env:
            return env
        v = VISION_MODELS.get(self.provider)
        return self.model if v is None else v

    @property
    def supports_vision(self) -> bool:
        return bool(self.available and self.vision_model)

    def complete_vision(self, system: str, user: str, images: list, max_tokens: int = 2500) -> str:
        """Like complete(), with photos: images = [(mime_type, bytes), ...] (jpeg or png)."""
        if not self.available:
            raise LLMError("no API key is set (GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY or "
                           "ANTHROPIC_API_KEY)")
        if not images:
            raise LLMError("no image was given")
        key_var, url, _default, wire = PROVIDERS[self.provider]
        enc = [(mime, base64.b64encode(data).decode("ascii")) for mime, data in images]
        if wire == "anthropic":
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
            content = [{"type": "image", "source": {"type": "base64", "media_type": m, "data": b}}
                       for m, b in enc] + [{"type": "text", "text": user}]
            body = {"model": self.vision_model, "max_tokens": max_tokens, "system": system,
                    "messages": [{"role": "user", "content": content}]}
        else:
            headers = {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": f"data:{m};base64,{b}"}} for m, b in enc]
            budget = max_tokens * 3 if self.provider == "gemini" else max_tokens
            body = {"model": self.vision_model, "max_tokens": budget, "temperature": 0.0,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": content}]}
        return self._send(url, headers, body, wire)


def main():
    """python -m src.inference.llm   -> sends one tiny question and prints the answer or the error"""
    try:
        c = LLMClient()
        print("provider / model:", c.describe())
        if not c.available:
            print("Set one key first, e.g. in PowerShell:  $env:GEMINI_API_KEY = \"your-key\"")
            return
        print("answer:", c.complete("Answer in one short sentence.",
                                    "What is the time complexity of binary search?", 200))
    except LLMError as e:
        print("FAILED:", e)


if __name__ == "__main__":
    main()