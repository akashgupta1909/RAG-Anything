"""Retrieval: hybrid search in Qdrant, then cross-encoder reranking.

``Retriever.search`` is deliberately a small, self-contained capability
(query + optional domain filter -> ranked chunks) so that a future agent can
expose it directly as a tool.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rag.config import RetrievalConfig, RetrievalMode
from rag.embeddings import DenseEncoder, ImageEncoder, SparseEncoder
from rag.models import Diagram, RetrievedChunk
from rag.retrieval.reranker import CrossEncoderReranker
from rag.vectordb import QdrantStore

# CLIP's text tower reads at most 77 tokens; keep queries comfortably short.
_CLIP_QUERY_CHARS = 200


class Retriever:
    def __init__(
        self,
        config: RetrievalConfig,
        store: QdrantStore,
        dense: DenseEncoder,
        sparse: SparseEncoder,
        reranker: CrossEncoderReranker | None,
    ) -> None:
        self.config = config
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker

    def search(
        self,
        query: str,
        *,
        domains: Sequence[str] | None = None,
        top_k: int | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most relevant chunks for ``query``.

        ``mode``/``rerank`` default to the configured values; overriding them is
        how the evaluation harness runs ablations (dense vs BM25 vs hybrid).
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        mode = mode or self.config.mode
        top_k = top_k or self.config.top_k
        use_rerank = (
            self.config.rerank.enabled if rerank is None else rerank
        ) and self.reranker is not None

        candidates = self.store.search(
            dense=self.dense.embed_query(query)
            if mode in {"hybrid", "dense"}
            else None,
            sparse=self.sparse.embed_query(query)
            if mode in {"hybrid", "sparse"}
            else None,
            mode=mode,
            domains=domains,
            limit=max(self.config.candidates, top_k) if use_rerank else top_k,
            prefetch_limit=self.config.prefetch_limit,
        )
        candidates = [c.model_copy(update={"query": query}) for c in candidates]
        if use_rerank and self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]


class DiagramSearch:
    """Multimodal search over indexed diagrams with CLIP (text -> image, image -> image)."""

    def __init__(self, store: QdrantStore, encoder: ImageEncoder) -> None:
        self.store = store
        self.encoder = encoder

    def by_text(
        self, text: str, *, domains: Sequence[str] | None = None, limit: int = 5
    ) -> list[Diagram]:
        vector = self.encoder.embed_text(text.strip()[:_CLIP_QUERY_CHARS])
        return self.store.search_images(vector, domains, limit)

    def by_image(
        self,
        image_path: str | Path,
        *,
        domains: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[Diagram]:
        vector = self.encoder.embed_images([image_path])[0]
        return self.store.search_images(vector, domains, limit)

    def for_chunks(self, chunks: Sequence[RetrievedChunk], limit: int) -> list[Diagram]:
        """Diagrams embedded in the given chunks, in chunk order."""
        image_ids = list(dict.fromkeys(i for c in chunks for i in c.image_ids))
        return self.store.get_images(image_ids)[:limit]
