"""Semantic search over the movie dataset using sentence embeddings."""

import hashlib
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

from config import (
    CACHE_DIR,
    CHUNK_EMBEDDINGS_FILE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MOVIE_EMBEDDINGS_FILE,
)
from lib.chunking import semantic_chunk
from lib.dataset import Movie, load_movies

Embedding = NDArray[np.float32]
"""One embedding vector (1-D), or a matrix of them with one row per text (2-D)."""

CacheArrays = dict[str, NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class SearchResult:
    score: float
    movie: Movie


def _require_text(text: str) -> None:
    if not text.strip():
        raise ValueError("text must not be empty or whitespace")


def _document_text(movie: Movie) -> str:
    return f"{movie['title']}:{movie['description']}"


def _fingerprint(documents: list[Movie]) -> str:
    """Hash of the documents' ids and text, used to detect stale caches."""
    digest = hashlib.sha256()
    for movie in documents:
        record = f"{movie['id']}\x1f{movie['title']}\x1f{movie['description']}\x1e"
        digest.update(record.encode())
    return digest.hexdigest()


def _top_k(scores: NDArray[np.float32], limit: int) -> list[int]:
    """Indices of the ``limit`` highest scores, best first; ties keep index order."""
    return cast(list[int], np.argsort(-scores, kind="stable")[:limit].tolist())


def _save_cache(path: Path, arrays: CacheArrays) -> None:
    """Write ``arrays`` to ``path`` atomically, so a crash never leaves a partial file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = path.with_name(f"{path.stem}.tmp.npz")
    np.savez(tmp_file, allow_pickle=False, **arrays)
    tmp_file.replace(path)


def _load_cache(path: Path, expected: dict[str, str]) -> CacheArrays | None:
    """Return the arrays in ``path`` if its string entries match ``expected``.

    Returns ``None`` when the cache is missing, unreadable, from an older format,
    or was built from different inputs (data, model or settings).
    """
    try:
        with np.load(path) as cache:
            arrays: CacheArrays = {name: cache[name] for name in cache.files}
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    for name, value in expected.items():
        if name not in arrays or str(arrays[name]) != value:
            return None
    return arrays


class SemanticSearch:
    """Embeds movies with a sentence-transformer model and ranks them by cosine similarity."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.documents: list[Movie] = []
        # One L2-normalised row per document, so a dot product is the cosine similarity.
        self.embeddings: Embedding | None = None

    @cached_property
    def model(self) -> SentenceTransformer:
        """The embedding model, loaded on first use (loading is slow)."""
        # The third-party loader can fail in many ways (network, disk, ...).
        try:
            return self._load_model()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {e}"
            ) from e

    def _load_model(self) -> SentenceTransformer:
        # Hide transformers' "Loading weights" bar; encode()'s own bar is unaffected.
        hf_logging.disable_progress_bar()
        try:
            # Use the locally cached copy if there is one: skips the Hugging Face Hub
            # update checks, which cost 1-2 s of network round-trips on every run.
            return SentenceTransformer(self.model_name, local_files_only=True)
        except OSError:
            # Not cached yet (first run): download it from the Hub.
            return SentenceTransformer(self.model_name)

    def _encode(
        self, texts: str | list[str], *, show_progress: bool = False
    ) -> Embedding:
        embeddings = self.model.encode(  # pyright: ignore[reportUnknownMemberType]
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return cast(Embedding, embeddings)

    def generate_embedding(self, text: str) -> Embedding:
        """Return the normalised embedding vector for ``text``."""
        _require_text(text)
        return self._encode(text)

    def _cache_key(self, documents: list[Movie]) -> dict[str, str]:
        return {"model_name": self.model_name, "fingerprint": _fingerprint(documents)}

    def build_embeddings(self, documents: list[Movie]) -> Embedding:
        """Embed every document and save the result to the cache."""
        self.documents = documents
        self.embeddings = self._encode(
            [_document_text(movie) for movie in documents], show_progress=True
        )
        _save_cache(
            MOVIE_EMBEDDINGS_FILE,
            {
                "embeddings": self.embeddings,
                **{k: np.array(v) for k, v in self._cache_key(documents).items()},
            },
        )
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> Embedding:
        """Load cached embeddings for ``documents``, rebuilding them if missing or stale."""
        cache = _load_cache(MOVIE_EMBEDDINGS_FILE, self._cache_key(documents))
        if cache is None:
            return self.build_embeddings(documents)
        self.documents = documents
        self.embeddings = cast(Embedding, cache["embeddings"])
        return self.embeddings

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Return the ``limit`` documents most similar to ``query``, best first."""
        if self.embeddings is None:
            raise RuntimeError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        # Rows are unit length, so the dot product is the cosine similarity.
        scores = self.embeddings @ self.generate_embedding(query)
        return [
            SearchResult(float(scores[i]), self.documents[i])
            for i in _top_k(scores, limit)
        ]


class ChunkedSemanticSearch(SemanticSearch):
    """Embeds each movie description as overlapping sentence chunks.

    A movie is scored by its best-matching chunk, so a query that matches one
    part of a long description is not diluted by the rest of it.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        super().__init__(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Row i of chunk_embeddings is chunk number chunk_indices[i] of the movie
        # at self.documents[chunk_movie_indices[i]].
        self.chunk_embeddings: Embedding | None = None
        self.chunk_movie_indices: NDArray[np.intp] | None = None
        self.chunk_indices: NDArray[np.intp] | None = None

    def _cache_key(self, documents: list[Movie]) -> dict[str, str]:
        return {
            **super()._cache_key(documents),
            "chunking": f"sentences size={self.chunk_size} overlap={self.chunk_overlap}",
        }

    def build_chunk_embeddings(self, documents: list[Movie]) -> Embedding:
        """Chunk and embed every description, then save the result to the cache."""
        chunks: list[str] = []
        movie_indices: list[int] = []
        chunk_indices: list[int] = []
        for movie_idx, movie in enumerate(documents):
            movie_chunks = semantic_chunk(
                movie["description"], self.chunk_size, self.chunk_overlap
            )
            chunks.extend(movie_chunks)
            movie_indices.extend([movie_idx] * len(movie_chunks))
            chunk_indices.extend(range(len(movie_chunks)))

        self.documents = documents
        self.chunk_embeddings = self._encode(chunks, show_progress=True)
        self.chunk_movie_indices = np.array(movie_indices, dtype=np.intp)
        self.chunk_indices = np.array(chunk_indices, dtype=np.intp)
        _save_cache(
            CHUNK_EMBEDDINGS_FILE,
            {
                "embeddings": self.chunk_embeddings,
                "movie_indices": self.chunk_movie_indices,
                "chunk_indices": self.chunk_indices,
                **{k: np.array(v) for k, v in self._cache_key(documents).items()},
            },
        )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> Embedding:
        """Load cached chunk embeddings, rebuilding them if missing or stale."""
        cache = _load_cache(CHUNK_EMBEDDINGS_FILE, self._cache_key(documents))
        if cache is None:
            return self.build_chunk_embeddings(documents)
        self.documents = documents
        self.chunk_embeddings = cast(Embedding, cache["embeddings"])
        self.chunk_movie_indices = cast(NDArray[np.intp], cache["movie_indices"])
        self.chunk_indices = cast(NDArray[np.intp], cache["chunk_indices"])
        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Return the ``limit`` movies whose best chunk is most similar to ``query``."""
        if self.chunk_embeddings is None or self.chunk_movie_indices is None:
            raise RuntimeError(
                "No chunk embeddings loaded. "
                "Call `load_or_create_chunk_embeddings` first."
            )
        # Rows are unit length, so the dot product is the cosine similarity.
        chunk_scores = self.chunk_embeddings @ self.generate_embedding(query)

        # Score each movie by its best chunk; movies without chunks stay at -inf.
        movie_scores = np.full(len(self.documents), -np.inf, dtype=np.float32)
        np.maximum.at(movie_scores, self.chunk_movie_indices, chunk_scores)

        return [
            SearchResult(float(movie_scores[i]), self.documents[i])
            for i in _top_k(movie_scores, limit)
            if np.isfinite(movie_scores[i])
        ]


def _load_movie_search() -> SemanticSearch:
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(load_movies())
    return semantic_search


def _load_chunked_search() -> ChunkedSemanticSearch:
    chunked_search = ChunkedSemanticSearch()
    chunked_search.load_or_create_chunk_embeddings(load_movies())
    return chunked_search


def verify_model() -> SentenceTransformer:
    """Load the embedding model and return it, raising if it cannot be loaded."""
    return SemanticSearch().model


def embed_text(text: str) -> Embedding:
    """Return the embedding vector for ``text``."""
    _require_text(text)  # fail fast, before the slow model load
    return SemanticSearch().generate_embedding(text)


def verify_embeddings() -> Embedding:
    """Load (or build) the movie embeddings and return the embedding matrix."""
    embeddings = _load_movie_search().embeddings
    assert embeddings is not None  # set by load_or_create_embeddings
    return embeddings


def search_movies(query: str, limit: int = 5) -> list[SearchResult]:
    """Return the ``limit`` movies most semantically similar to ``query``."""
    _require_text(query)  # fail fast, before loading movies and the model
    return _load_movie_search().search(query, limit)


def embed_chunks() -> Embedding:
    """Load (or build) the chunk embeddings for all movies and return them."""
    embeddings = _load_chunked_search().chunk_embeddings
    assert embeddings is not None  # set by load_or_create_chunk_embeddings
    return embeddings


def search_chunked(query: str, limit: int = 5) -> list[SearchResult]:
    """Return the ``limit`` movies whose best chunk matches ``query`` most closely."""
    _require_text(query)  # fail fast, before loading movies and the model
    return _load_chunked_search().search_chunks(query, limit)
