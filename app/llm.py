"""Model access: open-weight models on Groq, OpenRouter as fallback.

Both expose an OpenAI-compatible /chat/completions endpoint, so one small
client covers both. Configuration comes from environment variables (see
.env.example). Nothing here is ever logged with keys.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


class LLMError(Exception):
    """Every configured provider failed."""


def load_env(path: Path | None = None) -> None:
    """Minimal .env loader (avoids a python-dotenv dependency)."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


PROVIDERS = [
    {"name": "groq", "base": "https://api.groq.com/openai/v1",
     "key_env": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
     "default_model": "openai/gpt-oss-120b"},
    {"name": "openrouter", "base": "https://openrouter.ai/api/v1",
     "key_env": "OPENROUTER_API_KEY", "model_env": "OPENROUTER_MODEL",
     "default_model": "openai/gpt-oss-120b"},
]


class LLMClient:
    mode = "live"

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def configured(self) -> list[dict]:
        out = []
        for p in PROVIDERS:
            key = os.environ.get(p["key_env"], "").strip()
            if key:
                out.append({**p, "key": key,
                            "model": os.environ.get(p["model_env"], p["default_model"])})
        return out

    def describe(self) -> list[str]:
        return [f"{p['name']}:{p['model']}" for p in self.configured()]

    def complete(self, system: str, user: str, stage: str = "") -> tuple[str, str, str]:
        """Return (text, provider, model). Tries each provider in order,
        retrying briefly on rate limits and server errors."""
        providers = self.configured()
        if not providers:
            raise LLMError("No API key configured (set GROQ_API_KEY or OPENROUTER_API_KEY).")
        errors = []
        for p in providers:
            for attempt in range(2):
                try:
                    r = requests.post(
                        f"{p['base']}/chat/completions",
                        headers={"Authorization": f"Bearer {p['key']}",
                                 "Content-Type": "application/json"},
                        json={"model": p["model"], "temperature": 0,
                              "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": user}]},
                        timeout=self.timeout,
                    )
                except requests.RequestException as exc:
                    errors.append(f"{p['name']}: {exc.__class__.__name__}")
                    time.sleep(0.8)
                    continue
                if r.status_code == 200:
                    try:
                        text = r.json()["choices"][0]["message"]["content"] or ""
                    except Exception:
                        errors.append(f"{p['name']}: unexpected response shape")
                        break
                    return text, p["name"], p["model"]
                errors.append(f"{p['name']}: HTTP {r.status_code}")
                if r.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                    time.sleep(1.5)
                    continue
                break  # auth errors etc: go to next provider
        raise LLMError("All model providers failed (" + "; ".join(errors) + ").")


def get_client():
    """Live client if a key exists (and mock not forced), else scripted mock."""
    from .mock import MockLLM
    load_env()
    if os.environ.get("VERBATIM_MOCK", "").strip() in ("1", "true", "yes"):
        return MockLLM()
    live = LLMClient()
    return live if live.configured() else MockLLM()
