"""Shared fixtures: fake LLM and fake encoders, so tests need no network or models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag.config import DomainConfig, Settings, SourceConfig
from rag.embeddings import SparseVector
from rag.llm import LLMError, Messages
from rag.models import RetrievedChunk


def make_chunk(
    chunk_id: str,
    *,
    domain: str = "python",
    url: str | None = None,
    text: str = "text",
    score: float = 1.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"{domain}/src/{chunk_id}",
        domain=domain,
        title=f"Title {chunk_id}",
        url=url or f"https://example.com/{chunk_id}",
        heading_path=["Section"],
        text=text,
        score=score,
    )


class FakeLLM:
    """Returns queued replies (or a reply chosen by a function of the messages)."""

    def __init__(self, replies: list[str | Exception] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[Messages] = []
        self.available = ["fake"]

    def _next(self, messages: Messages) -> str:
        self.calls.append(messages)
        if not self.replies:
            raise LLMError("no fake replies left")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def complete(self, messages: Messages, **_: Any) -> str:
        return self._next(messages)

    def complete_json(self, messages: Messages, model: Any, **_: Any) -> Any:
        from rag.llm import parse_json_model

        return parse_json_model(self._next(messages), model)

    def stream(self, messages: Messages, **_: Any) -> Iterator[str]:
        yield from self._next(messages).split(" ")


class FakeDense:
    """Deterministic bag-of-words vectors: texts sharing words are similar."""

    dimension = 32

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeSparse:
    def _vec(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 100_000
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return SparseVector(indices=list(counts), values=list(counts.values()))

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._vec(text)


class FakeImage:
    dimension = 8

    def embed_images(self, paths: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 7 for _ in paths]

    def embed_text(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 7


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    source = SourceConfig(
        name="docs", kind="local", location=str(tmp_path / "src"), format="markdown"
    )
    settings = Settings(
        domains=[
            DomainConfig(name="python", description="Python docs", sources=[source]),
            DomainConfig(
                name="kubernetes", description="Kubernetes docs", sources=[source]
            ),
        ],
    )
    settings.vectordb.path = tmp_path / "qdrant"
    settings.app.log_dir = tmp_path / "logs"
    return settings


def plan_json(
    domains: list[str], sub_queries: list[str], diagrams: bool = False
) -> str:
    return json.dumps(
        {"domains": domains, "sub_queries": sub_queries, "needs_diagrams": diagrams}
    )
