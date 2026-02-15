from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class StubLLM:
    def complete(self, prompt: str) -> str:
        return (
            "Stub response (set OPENAI_API_KEY for real model calls).\n\n"
            "Received prompt summary:\n"
            f"{prompt[:800]}"
        )


class OpenAILLM:
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


def build_default_llm() -> LLM:
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAILLM()
        except Exception:
            return StubLLM()
    return StubLLM()

