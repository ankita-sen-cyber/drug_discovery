from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import RAGConfig


WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text)]


def _cosine_sim(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str
    embedding: Counter[str]


class LocalRAG:
    def __init__(self, config: RAGConfig):
        self.config = config
        self._chunks: list[Chunk] = []

    def index_dir(self, docs_dir: Path) -> None:
        files = list(docs_dir.rglob("*.txt")) + list(docs_dir.rglob("*.md"))
        for file in files:
            content = file.read_text(encoding="utf-8", errors="ignore")
            for text_chunk in self._chunk_text(content):
                self._chunks.append(
                    Chunk(
                        source=str(file),
                        text=text_chunk,
                        embedding=Counter(_tokenize(text_chunk)),
                    )
                )

    def query(self, question: str, top_k: int | None = None) -> list[Chunk]:
        k = top_k if top_k is not None else self.config.top_k
        q = Counter(_tokenize(question))
        ranked = sorted(
            self._chunks,
            key=lambda c: _cosine_sim(q, c.embedding),
            reverse=True,
        )
        return ranked[:k]

    def _chunk_text(self, text: str) -> Iterable[str]:
        size = self.config.chunk_size_chars
        overlap = self.config.overlap_chars
        if size <= overlap:
            raise ValueError("chunk_size_chars must be greater than overlap_chars")
        i = 0
        while i < len(text):
            yield text[i : i + size].strip()
            i += size - overlap

