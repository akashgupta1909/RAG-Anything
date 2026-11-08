"""Cross-encoder reranking: score each (query, chunk) pair jointly.

Bi-encoder and BM25 retrieval are fast but score query and document
independently; a cross-encoder reads them together and is far more precise,
so it is applied to a small candidate set only.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.embeddings.encoders import load_model
from rag.models import RetrievedChunk
from rag.utils import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir

    @cached_property
    def _model(self) -> TextCrossEncoder:
        logger.info("Loading reranker %s", self.model_name)
        return load_model(TextCrossEncoder, self.model_name, self.cache_dir)

    def load(self) -> None:
        """Load the model now instead of on the first query."""
        _ = self._model

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks by cross-encoder score, best first."""
        if not chunks:
            return []
        documents = [f"{c.breadcrumb}\n{c.text}" for c in chunks]
        scores = list(self._model.rerank(query, documents))
        ranked = sorted(
            zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [
            chunk.model_copy(update={"rerank_score": float(score)})
            for chunk, score in ranked[:top_k]
        ]
