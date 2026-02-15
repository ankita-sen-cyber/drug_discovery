from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RAGConfig:
    chunk_size_chars: int = 1500
    overlap_chars: int = 250
    top_k: int = 6


@dataclass(frozen=True)
class AppConfig:
    docs_dir: Path
    rag: RAGConfig = RAGConfig()

