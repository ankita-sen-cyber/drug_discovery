from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class StubLLM:
    provider = "stub"

    def complete(self, prompt: str) -> str:
        return (
            "Stub response (set OPENAI_API_KEY for real model calls).\n\n"
            "Received prompt summary:\n"
            f"{prompt[:800]}"
        )


class OpenAILLM:
    provider = "openai"

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def complete(self, prompt: str) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return resp.output_text


class OllamaLLM:
    provider = "ollama"

    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body.get("response", "")
        if not text:
            raise RuntimeError(f"Ollama returned empty response payload: {body}")
        return text


def get_llm_provider_name(llm: LLM) -> str:
    return getattr(llm, "provider", llm.__class__.__name__.lower())


def build_default_llm() -> LLM:
    # Prefer explicit provider selection when set.
    provider = os.getenv("KDENSE_LLM_PROVIDER", "").strip().lower()
    if provider == "ollama":
        try:
            return OllamaLLM()
        except Exception:
            return StubLLM()
    if provider == "openai":
        try:
            return OpenAILLM()
        except Exception:
            return StubLLM()

    # Auto-detect: OpenAI key first, then Ollama, then stub.
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAILLM()
        except Exception:
            pass
    if os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_BASE_URL"):
        try:
            return OllamaLLM()
        except (urllib.error.URLError, TimeoutError, Exception):
            return StubLLM()
    return StubLLM()
