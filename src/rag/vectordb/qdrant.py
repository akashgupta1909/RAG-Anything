"""Qdrant vector store: hybrid (dense + BM25) text chunks and CLIP image vectors.

Runs against a Qdrant server when ``url`` is set, or embedded in-process
(``path``) for zero-setup local development.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from rag.config import RetrievalMode, VectorDBConfig
from rag.embeddings import SparseVector
from rag.models import Chunk, Diagram, ImageRef, RetrievedChunk
from rag.utils import get_logger, stable_uuid

logger = get_logger(__name__)

DENSE = "dense"
SPARSE = "bm25"
IMAGE = "clip"


class VectorStoreError(RuntimeError):
    """The vector store is unreachable or misconfigured."""


def _domain_filter(domains: Sequence[str] | None) -> qm.Filter | None:
    if not domains:
        return None
    return qm.Filter(
        must=[qm.FieldCondition(key="domain", match=qm.MatchAny(any=list(domains)))]
    )


class QdrantStore:
    def __init__(self, config: VectorDBConfig) -> None:
        self.config = config
        self.text_collection = config.text_collection
        self.image_collection = config.image_collection
        if config.url:
            self.client = QdrantClient(url=config.url, api_key=config.api_key or None)
            self.location = config.url
        else:
            config.path.mkdir(parents=True, exist_ok=True)
            try:
                self.client = QdrantClient(path=str(config.path))
            except RuntimeError as e:  # embedded storage is single-process
                raise VectorStoreError(
                    f"{e}. Embedded Qdrant allows one process at a time: stop the "
                    "API server, or run Qdrant as a server (docker compose up -d qdrant)."
                ) from e
            self.location = str(config.path)

    # -- schema ------------------------------------------------------------
    def ensure_collections(
        self, dense_dim: int, image_dim: int, *, recreate: bool = False
    ) -> None:
        """Create the collections (and payload indexes) if they do not exist."""
        for name, vectors, sparse in (
            (
                self.text_collection,
                {DENSE: qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)},
                {SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            ),
            (
                self.image_collection,
                {IMAGE: qm.VectorParams(size=image_dim, distance=qm.Distance.COSINE)},
                None,
            ),
        ):
            if recreate and self.client.collection_exists(name):
                self.client.delete_collection(name)
            if self.client.collection_exists(name):
                self._check_dimension(name, vectors)
                continue
            logger.info("Creating collection %s", name)
            self.client.create_collection(
                name, vectors_config=vectors, sparse_vectors_config=sparse
            )
            if self.config.url:  # payload indexes are a no-op in embedded mode
                for field in ("domain", "doc_id", "image_id"):
                    self.client.create_payload_index(
                        name, field, qm.PayloadSchemaType.KEYWORD
                    )
                self.client.create_payload_index(
                    name, "chunk_index", qm.PayloadSchemaType.INTEGER
                )

    def _check_dimension(self, name: str, vectors: dict[str, qm.VectorParams]) -> None:
        existing = self.client.get_collection(name).config.params.vectors
        if not isinstance(existing, dict):
            return
        for key, params in vectors.items():
            if key in existing and existing[key].size != params.size:
                raise VectorStoreError(
                    f"collection {name!r} stores {existing[key].size}-d '{key}' vectors "
                    f"but the model produces {params.size}-d; re-ingest with --recreate"
                )

    # -- writes ------------------------------------------------------------
    def document_hashes(self) -> dict[str, tuple[str, str]]:
        """``doc_id -> (domain, content hash)`` of every indexed document."""
        if not self.client.collection_exists(self.text_collection):
            return {}
        hashes: dict[str, tuple[str, str]] = {}
        offset: Any = None
        first_chunks = qm.Filter(
            must=[qm.FieldCondition(key="chunk_index", match=qm.MatchValue(value=0))]
        )
        while True:
            points, offset = self.client.scroll(
                self.text_collection,
                scroll_filter=first_chunks,
                with_payload=["doc_id", "doc_hash", "domain"],
                limit=1000,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                hashes[payload["doc_id"]] = (payload["domain"], payload["doc_hash"])
            if offset is None:
                return hashes

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        if not doc_ids:
            return
        selector = qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="doc_id", match=qm.MatchAny(any=list(doc_ids))
                    )
                ]
            )
        )
        for collection in (self.text_collection, self.image_collection):
            self.client.delete(collection, points_selector=selector, wait=True)

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[list[float]],
        sparse: Sequence[SparseVector],
        doc_hashes: dict[str, str],
    ) -> None:
        points = [
            qm.PointStruct(
                id=chunk.chunk_id,
                vector={
                    DENSE: d,
                    SPARSE: qm.SparseVector(indices=s.indices, values=s.values),
                },
                payload={**chunk.model_dump(), "doc_hash": doc_hashes[chunk.doc_id]},
            )
            for chunk, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        self._upsert(self.text_collection, points)

    def upsert_images(
        self,
        images: Sequence[tuple[ImageRef, str, str, str]],
        vectors: Sequence[list[float]],
    ) -> None:
        """Index ``(image, domain, doc_id, doc_url)`` tuples with their CLIP vectors."""
        points = [
            qm.PointStruct(
                # One point per (document, image), so deleting a document never
                # removes a diagram another document still references.
                id=stable_uuid(f"{doc_id}#img:{image.image_id}"),
                vector={IMAGE: vector},
                payload={
                    "image_id": image.image_id,
                    "url": image.url,
                    "caption": image.caption,
                    "heading_path": image.heading_path,
                    "domain": domain,
                    "doc_id": doc_id,
                    "doc_url": doc_url,
                },
            )
            for (image, domain, doc_id, doc_url), vector in zip(
                images, vectors, strict=True
            )
        ]
        self._upsert(self.image_collection, points)

    def _upsert(self, collection: str, points: list[qm.PointStruct]) -> None:
        size = self.config.upsert_batch_size
        for start in range(0, len(points), size):
            self.client.upsert(
                collection, points=points[start : start + size], wait=True
            )

    # -- reads -------------------------------------------------------------
    def search(
        self,
        *,
        dense: list[float] | None,
        sparse: SparseVector | None,
        mode: RetrievalMode,
        domains: Sequence[str] | None,
        limit: int,
        prefetch_limit: int,
    ) -> list[RetrievedChunk]:
        """Dense, sparse (BM25) or hybrid search; hybrid fuses both with RRF in Qdrant."""
        query_filter = _domain_filter(domains)
        sparse_query = (
            qm.SparseVector(indices=sparse.indices, values=sparse.values)
            if sparse
            else None
        )
        if mode == "hybrid":
            if dense is None or sparse_query is None:
                raise ValueError("hybrid search needs both dense and sparse vectors")
            response = self.client.query_points(
                self.text_collection,
                prefetch=[
                    qm.Prefetch(
                        query=dense,
                        using=DENSE,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    qm.Prefetch(
                        query=sparse_query,
                        using=SPARSE,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            query = dense if mode == "dense" else sparse_query
            if query is None:
                raise ValueError(f"{mode} search needs a {mode} vector")
            response = self.client.query_points(
                self.text_collection,
                query=query,
                using=DENSE if mode == "dense" else SPARSE,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        return [
            RetrievedChunk(**{**(p.payload or {}), "score": p.score})
            for p in response.points
        ]

    def search_images(
        self, vector: list[float], domains: Sequence[str] | None, limit: int
    ) -> list[Diagram]:
        response = self.client.query_points(
            self.image_collection,
            query=vector,
            using=IMAGE,
            query_filter=_domain_filter(domains),
            limit=limit * 3,  # the same image can be indexed for several documents
            with_payload=True,
        )
        diagrams: dict[str, Diagram] = {}
        for p in response.points:
            diagram = Diagram(**{**(p.payload or {}), "score": p.score})
            diagrams.setdefault(diagram.image_id, diagram)
        return list(diagrams.values())[:limit]

    def get_images(self, image_ids: Sequence[str]) -> list[Diagram]:
        """Fetch diagrams by image id, in the order given (unknown ids are skipped)."""
        if not image_ids:
            return []
        points, _ = self.client.scroll(
            self.image_collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="image_id", match=qm.MatchAny(any=list(image_ids))
                    )
                ]
            ),
            with_payload=True,
            limit=len(image_ids) * 5,
        )
        by_id: dict[str, Diagram] = {}
        for p in points:
            diagram = Diagram(**(p.payload or {}))
            by_id.setdefault(diagram.image_id, diagram)
        return [by_id[i] for i in image_ids if i in by_id]

    def stats(self) -> dict[str, Any]:
        """Point counts per collection and per domain."""
        stats: dict[str, Any] = {"location": self.location, "collections": {}}
        for name in (self.text_collection, self.image_collection):
            if not self.client.collection_exists(name):
                stats["collections"][name] = None
                continue
            stats["collections"][name] = self.client.count(name, exact=True).count
        hashes = self.document_hashes()
        per_domain: dict[str, int] = {}
        for domain, _ in hashes.values():
            per_domain[domain] = per_domain.get(domain, 0) + 1
        stats["documents"] = per_domain
        return stats

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
        except Exception:  # any transport error means unhealthy
            return False
        return True

    def close(self) -> None:
        self.client.close()
