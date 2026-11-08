"""Ingestion: fetch -> load -> chunk -> embed -> upsert, incrementally.

Each document's content hash (text + chunker settings + embedding models) is
stored with its chunks. Re-running ingestion therefore only re-embeds
documents that changed, and removes documents that disappeared from a source.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rag.chunking import MarkdownChunker
from rag.config import DomainConfig, Settings
from rag.embeddings import DenseEncoder, ImageEncoder, SparseEncoder
from rag.ingestion.loaders import DocumentLoader
from rag.ingestion.sources import fetch_source, iter_source_files
from rag.models import Chunk, Document, ImageRef
from rag.utils import get_logger, sha256
from rag.vectordb import QdrantStore

logger = get_logger(__name__)

ProgressCallback = Callable[[str, int, int], None]
"""Called as ``on_progress(stage, done, total)``."""

_DOC_BATCH = 32  # documents embedded per batch


@dataclass
class IngestionReport:
    documents_seen: int = 0
    documents_indexed: int = 0
    documents_unchanged: int = 0
    documents_deleted: int = 0
    chunks_indexed: int = 0
    images_indexed: int = 0
    per_domain: dict[str, int] = field(default_factory=dict)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        store: QdrantStore,
        dense: DenseEncoder,
        sparse: SparseEncoder,
        images: ImageEncoder,
    ) -> None:
        self.settings = settings
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.images = images
        self.chunker = MarkdownChunker(
            settings.chunking.max_chars, settings.chunking.min_chars
        )

    def _content_hash(self, doc: Document) -> str:
        emb = self.settings.embeddings
        parts = [
            doc.title,
            doc.url,
            doc.text,
            self.chunker.signature,
            emb.dense_model,
            emb.sparse_model,
            emb.image_model if doc.images else "",
        ]
        return sha256("\x1f".join(parts))

    def run(
        self,
        domains: Sequence[str] | None = None,
        *,
        refresh_sources: bool = False,
        recreate: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionReport:
        """Ingest ``domains`` (default: all configured domains)."""
        selected = (
            [self.settings.domain(d) for d in domains]
            if domains
            else self.settings.domains
        )
        self.store.ensure_collections(
            self.dense.dimension, self.images.dimension, recreate=recreate
        )
        existing = self.store.document_hashes()
        report = IngestionReport()

        for domain in selected:
            docs = self._load_domain(domain, refresh_sources)
            report.documents_seen += len(docs)
            report.per_domain[domain.name] = len(docs)

            current_ids = {d.doc_id for d in docs}
            stale = [
                doc_id
                for doc_id, (doc_domain, _) in existing.items()
                if doc_domain == domain.name and doc_id not in current_ids
            ]
            if stale:
                logger.info(
                    "Removing %d deleted documents from %s", len(stale), domain.name
                )
                self.store.delete_documents(stale)
                report.documents_deleted += len(stale)

            changed: list[tuple[Document, str]] = []
            for doc in docs:
                doc_hash = self._content_hash(doc)
                if existing.get(doc.doc_id, ("", ""))[1] == doc_hash:
                    report.documents_unchanged += 1
                else:
                    changed.append((doc, doc_hash))
            logger.info(
                "%s: %d documents, %d new or changed",
                domain.name,
                len(docs),
                len(changed),
            )
            for start in range(0, len(changed), _DOC_BATCH):
                batch = changed[start : start + _DOC_BATCH]
                chunks, images = self._index_batch(batch)
                report.documents_indexed += len(batch)
                report.chunks_indexed += chunks
                report.images_indexed += images
                if on_progress:
                    on_progress(
                        domain.name, min(start + _DOC_BATCH, len(changed)), len(changed)
                    )
        return report

    def _load_domain(self, domain: DomainConfig, refresh: bool) -> list[Document]:
        docs: list[Document] = []
        for source in domain.sources:
            root = fetch_source(
                source, self.settings.paths.raw_data_dir, refresh=refresh
            )
            loader = DocumentLoader(domain.name, source, root)
            try:
                for path, relative in iter_source_files(source, root):
                    try:
                        doc = loader.load(path, relative)
                    except (OSError, ValueError) as e:
                        logger.warning("Skipping %s: %s", relative, e)
                        continue
                    if doc is not None:
                        docs.append(doc)
            finally:
                loader.close()
        return docs

    def _index_batch(self, batch: list[tuple[Document, str]]) -> tuple[int, int]:
        """Replace the chunks and images of every document in ``batch``."""
        self.store.delete_documents([doc.doc_id for doc, _ in batch])

        chunks: list[Chunk] = []
        for doc, _ in batch:
            chunks.extend(self.chunker.chunk(doc))
        if chunks:
            texts = [c.embedding_text for c in chunks]
            self.store.upsert_chunks(
                chunks,
                self.dense.embed_documents(texts),
                self.sparse.embed_documents(texts),
                {doc.doc_id: doc_hash for doc, doc_hash in batch},
            )

        image_rows: list[tuple[ImageRef, str, str, str]] = [
            (image, doc.domain, doc.doc_id, doc.url)
            for doc, _ in batch
            for image in doc.images
        ]
        if image_rows:
            vectors = self.images.embed_images(
                [img.local_path for img, *_ in image_rows]
            )
            self.store.upsert_images(image_rows, vectors)
        return len(chunks), len(image_rows)
