"""Composition root: builds every service once from ``Settings``.

The CLI, the API and the evaluation harness all go through ``Container``, so
models are loaded once per process and components can be swapped in one place
(e.g. fakes in tests).
"""

from __future__ import annotations

from functools import cached_property

from rag.config import Settings
from rag.embeddings import DenseEncoder, ImageEncoder, SparseEncoder
from rag.ingestion import IngestionPipeline
from rag.llm import LLMClient
from rag.pipeline import ContextGrader, QueryPlanner, RAGPipeline
from rag.retrieval import CrossEncoderReranker, DiagramSearch, Retriever
from rag.vectordb import QdrantStore


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def store(self) -> QdrantStore:
        return QdrantStore(self.settings.vectordb)

    @cached_property
    def dense_encoder(self) -> DenseEncoder:
        emb = self.settings.embeddings
        return DenseEncoder(
            emb.dense_model, self.settings.paths.model_cache_dir, emb.batch_size
        )

    @cached_property
    def sparse_encoder(self) -> SparseEncoder:
        emb = self.settings.embeddings
        return SparseEncoder(
            emb.sparse_model, self.settings.paths.model_cache_dir, emb.batch_size
        )

    @cached_property
    def image_encoder(self) -> ImageEncoder:
        emb = self.settings.embeddings
        return ImageEncoder(
            emb.image_model, emb.image_text_model, self.settings.paths.model_cache_dir
        )

    @cached_property
    def reranker(self) -> CrossEncoderReranker:
        # Always available (the model loads lazily on first use), so a request
        # can opt in with rerank=True even when it is off by default.
        rerank = self.settings.retrieval.rerank
        return CrossEncoderReranker(rerank.model, self.settings.paths.model_cache_dir)

    @cached_property
    def retriever(self) -> Retriever:
        return Retriever(
            self.settings.retrieval,
            self.store,
            self.dense_encoder,
            self.sparse_encoder,
            self.reranker,
        )

    @cached_property
    def diagram_search(self) -> DiagramSearch:
        return DiagramSearch(self.store, self.image_encoder)

    @cached_property
    def llm(self) -> LLMClient:
        """Answer generation."""
        return LLMClient(self.settings.llm)

    @cached_property
    def fast_llm(self) -> LLMClient:
        """Planning and grading: short JSON outputs, so a small fast model."""
        fast = self.settings.llm.fast_providers
        return LLMClient(self.settings.llm, fast) if fast else self.llm

    @cached_property
    def judge_llm(self) -> LLMClient:
        """A different model family from the generator, for unbiased evaluation."""
        judge = self.settings.llm.judge
        return LLMClient(self.settings.llm, [judge]) if judge else self.llm

    @cached_property
    def pipeline(self) -> RAGPipeline:
        return RAGPipeline(
            retriever=self.retriever,
            diagrams=self.diagram_search,
            llm=self.llm,
            planner=QueryPlanner(
                self.fast_llm, self.settings.domains, self.settings.agent
            ),
            grader=ContextGrader(self.fast_llm),
            agent=self.settings.agent,
            generation=self.settings.generation,
        )

    @cached_property
    def ingestion(self) -> IngestionPipeline:
        return IngestionPipeline(
            self.settings,
            self.store,
            self.dense_encoder,
            self.sparse_encoder,
            self.image_encoder,
        )

    def warm_up(self) -> None:
        """Load all models up front (used by the API so the first request is fast)."""
        _ = self.dense_encoder.dimension
        self.sparse_encoder.embed_query("warm up")
        _ = self.image_encoder.dimension
        if self.settings.retrieval.rerank.enabled:
            self.reranker.load()  # load the cross-encoder only if it is used

    def close(self) -> None:
        if "store" in self.__dict__:
            self.store.close()
