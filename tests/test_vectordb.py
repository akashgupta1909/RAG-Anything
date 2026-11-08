"""Integration tests against a real (embedded) Qdrant, with fake encoders."""

from pathlib import Path
from typing import Any, cast

import pytest

from rag.config import Settings, SourceConfig
from rag.ingestion import IngestionPipeline
from rag.retrieval import Retriever
from rag.vectordb import QdrantStore
from tests.conftest import FakeDense, FakeImage, FakeSparse


@pytest.fixture
def corpus(tmp_path: Path, settings: Settings) -> Settings:
    py = tmp_path / "py"
    k8s = tmp_path / "k8s"
    py.mkdir()
    k8s.mkdir()
    (py / "asyncio.md").write_text(
        "# asyncio\n\n## Tasks\n\nasyncio gather runs coroutines concurrently. " * 3
    )
    (py / "gil.md").write_text(
        "# Threads\n\n## GIL\n\nThe global interpreter lock limits threads. " * 3
    )
    (k8s / "pods.md").write_text(
        "---\ntitle: Pods\n---\n## Pods\n\nA pod groups containers with shared network. "
        * 3
    )
    settings.domains[0].sources = [
        SourceConfig(
            name="py",
            kind="local",
            location=str(py),
            format="markdown",
            include=["*.md"],
        )
    ]
    settings.domains[1].sources = [
        SourceConfig(
            name="k8s",
            kind="local",
            location=str(k8s),
            format="markdown",
            include=["*.md"],
        )
    ]
    return settings


def _ingestion(settings: Settings, store: QdrantStore) -> IngestionPipeline:
    return IngestionPipeline(
        settings,
        store,
        cast(Any, FakeDense()),
        cast(Any, FakeSparse()),
        cast(Any, FakeImage()),
    )


def test_ingest_search_and_incremental_update(corpus: Settings, tmp_path: Path) -> None:
    store = QdrantStore(corpus.vectordb)
    try:
        report = _ingestion(corpus, store).run()
        assert report.documents_indexed == 3 and report.chunks_indexed >= 3

        retriever = Retriever(
            corpus.retrieval,
            store,
            cast(Any, FakeDense()),
            cast(Any, FakeSparse()),
            None,
        )
        for mode in ("hybrid", "dense", "sparse"):
            hits = retriever.search("asyncio gather coroutines", mode=mode, top_k=2)
            assert hits[0].title == "asyncio", mode

        k8s_only = retriever.search("asyncio gather coroutines", domains=["kubernetes"])
        assert {h.domain for h in k8s_only} == {"kubernetes"}

        # Unchanged documents are skipped; edited and deleted ones are handled.
        again = _ingestion(corpus, store).run()
        assert again.documents_unchanged == 3 and again.documents_indexed == 0

        (tmp_path / "py" / "gil.md").unlink()
        (tmp_path / "py" / "asyncio.md").write_text(
            "# asyncio\n\n## Queues\n\nasyncio Queue distributes work. " * 3
        )
        third = _ingestion(corpus, store).run()
        assert third.documents_deleted == 1 and third.documents_indexed == 1
        assert set(store.document_hashes()) == {
            "python/py/asyncio.md",
            "kubernetes/k8s/pods.md",
        }
    finally:
        store.close()
