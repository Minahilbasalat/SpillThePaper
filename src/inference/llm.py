"""Language-model client for AI-written solutions. Plain HTTPS through `requests`: nothing extra
to install, and several providers work (Gemini, Groq, Mistral, Anthropic).

Configuration = environment variables. NEVER put a key in the code, in a file that is committed,
or in a chat message.

    GEMINI_API_KEY   or  GROQ_API_KEY  or  MISTRAL_API_KEY  or  ANTHROPIC_API_KEY
    LLM_PROVIDER     optional: gemini | groq | mistral | anthropic
                     (default: the first of these whose key is set, in this order)
    LLM_VISION_MODEL optional: model that reads photos (default: Gemini/Anthropic read photos with
                     the normal model; Groq/Mistral use the model named in VISION_MODELS below)
    LLM_MODEL        optional: model name for the main provider (see the defaults below; model
                     names change, if a request says "model not found" copy a current name from
                     the provider's dashboard)
    GEMINI_MODEL / GROQ_MODEL / MISTRAL_MODEL / ANTHROPIC_MODEL
                     optional: model name of one provider (wins over LLM_MODEL for that provider)

If the main provider fails (quota, busy, wrong model) the client tries the next model of the same
provider (FALLBACK_MODELS, its own quota) and then every other provider whose key is set, so a
Groq key next to the Gemini key is a safety net.

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
               "gemini-3.8-flash", "openai"),
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
# Second choice of the SAME provider (each Gemini model has its own free quota)
FALLBACK_MODELS = {"gemini": ["gemini-3.5-flash-lite"], "groq": ["openai/gpt-oss-20b"]}
MODEL_ENV = {"gemini": "GEMINI_MODEL", "groq": "GROQ_MODEL", "mistral": "MISTRAL_MODEL",
             "anthropic": "ANTHROPIC_MODEL"}
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
        main = pick_provider() if self.provider else None
        generic = os.environ.get("LLM_MODEL") if self.provider == main else None   # LLM_MODEL = main provider only
        self.model = (model or (os.environ.get(MODEL_ENV[self.provider], "").strip() if self.provider else "")
                      or generic or default_model)
        self.timeout = timeout
        self._single = False               # True for the helper clients used inside the fallback chain

    @property
    def available(self) -> bool:
        return bool(self.provider and self.api_key)

    def describe(self) -> str:
        if not self.available:
            return "no API key set"
        extra = [f"{c.provider} / {c.model}" for c in self._chain()[1:]]
        return f"{self.provider} / {self.model}" + (f"  (then: {', '.join(extra)})" if extra else "")

    def _chain(self) -> list:
        """Clients to try in order: this one, the fallback model(s) of its provider, other providers with a key."""
        if self._single:
            return [self]
        out = [self]
        seen = {(self.provider, self.model)}

        def add(provider, key, model):
            if (provider, model) not in seen and key:
                seen.add((provider, model))
                c = LLMClient(provider=provider, api_key=key, model=model, timeout=self.timeout)
                c._single = True
                out.append(c)
        for m in FALLBACK_MODELS.get(self.provider, []):
            add(self.provider, self.api_key, m)
        for prov, spec in PROVIDERS.items():
            if prov != self.provider:
                key = os.environ.get(spec[0], "")
                add(prov, key, os.environ.get(MODEL_ENV[prov], "").strip() or spec[2])
                for m in FALLBACK_MODELS.get(prov, []):
                    add(prov, key, m)
        return out

    def _run(self, call):
        """Try every client of the chain; raise one readable LLMError if all fail."""
        errors = []
        for c in self._chain():
            try:
                return call(c)
            except LLMError as e:
                errors.append(f"{c.provider}/{c.model}: {e}")
        raise LLMError(errors[0] if len(errors) == 1 else
                       "every AI model failed - " + " | ".join(errors))

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
            if r.status_code in RETRY_STATUS and attempt < (1 if r.status_code == 429 else 2):
                try:
                    wait = float(r.headers.get("retry-after", 2 * (attempt + 1)))
                except ValueError:
                    wait = 2 * (attempt + 1)
                time.sleep(min(wait, MAX_WAIT_S))
                continue
            break
        if last.startswith(("401", "403")):
            msg = f"the API key was refused ({last[:3]}) - check the API key"
        elif last.startswith("404"):
            msg = "model not found (404) - set LLM_MODEL (or LLM_VISION_MODEL) to a current model name"
        elif last.startswith("429"):
            msg = "quota or rate limit reached (429) - wait a minute and try again"
        else:
            msg = f"request failed ({last})"
        raise LLMError(f"the {self.provider} {msg}")

    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        if not self.available:
            raise LLMError("no API key is set (GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY or "
                           "ANTHROPIC_API_KEY)")
        if self._single:
            return self._complete_one(system, user, max_tokens)
        return self._run(lambda c: c._complete_one(system, user, max_tokens))

    def _complete_one(self, system: str, user: str, max_tokens: int) -> str:
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
        if self._single:
            return self._vision_one(system, user, images, max_tokens)
        return self._run(lambda c: c._vision_one(system, user, images, max_tokens)
                         if c.supports_vision else (_ for _ in ()).throw(LLMError("no vision model")))

    def _vision_one(self, system: str, user: str, images: list, max_tokens: int) -> str:
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