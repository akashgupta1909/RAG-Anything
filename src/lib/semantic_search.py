"""Semantic search over the movie dataset using sentence embeddings."""

import zipfile
from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from config import CACHE_DIR, EMBEDDING_MODEL, MOVIE_EMBEDDINGS_FILE
from lib.search_util import Movie, load_movies

Embedding = NDArray[np.float32]
"""One embedding vector (1-D), or a matrix of them with one row per text (2-D)."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    score: float
    movie: Movie


def _require_text(text: str) -> None:
    if not text.strip():
        raise ValueError("text must not be empty or whitespace")


def _document_text(movie: Movie) -> str:
    return f"{movie['title']}:{movie['description']}"


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
        try:
            return SentenceTransformer(self.model_name)
        except Exception as e:  # third-party loader can fail in many ways (network, disk, ...)
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}") from e

    def _encode(self, texts: str | list[str], *, show_progress: bool = False) -> Embedding:
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

    def build_embeddings(self, documents: list[Movie]) -> Embedding:
        """Embed every document and save the result to the cache."""
        self.documents = documents
        self.embeddings = self._encode(
            [_document_text(movie) for movie in documents], show_progress=True
        )
        self._save_cache()
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> Embedding:
        """Load cached embeddings for ``documents``, rebuilding them if missing or stale."""
        cached = self._load_cache(documents)
        if cached is None:
            return self.build_embeddings(documents)
        self.documents = documents
        self.embeddings = cached
        return cached

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Return the ``limit`` documents most similar to ``query``, best first."""
        if self.embeddings is None:
            raise RuntimeError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores = self.embeddings @ query_embedding  # cosine similarity: rows are unit length
        top_indices = cast(list[int], np.argsort(-scores, kind="stable")[:limit].tolist())
        return [SearchResult(float(scores[i]), self.documents[i]) for i in top_indices]

    def _save_cache(self) -> None:
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = MOVIE_EMBEDDINGS_FILE.with_name(f"{MOVIE_EMBEDDINGS_FILE.stem}.tmp.npz")
        np.savez(
            tmp_file,
            embeddings=self.embeddings,
            doc_ids=np.array([movie["id"] for movie in self.documents]),
            model_name=np.array(self.model_name),
        )
        tmp_file.replace(MOVIE_EMBEDDINGS_FILE)

    def _load_cache(self, documents: list[Movie]) -> Embedding | None:
        """Return cached embeddings if they were built for ``documents`` with this model."""
        try:
            with np.load(MOVIE_EMBEDDINGS_FILE) as cache:
                embeddings = cast(Embedding, cache["embeddings"])
                doc_ids = cast(list[int], cache["doc_ids"].tolist())
                model_name = str(cache["model_name"])
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            return None  # missing, unreadable or old-format cache
        if model_name != self.model_name or doc_ids != [movie["id"] for movie in documents]:
            return None
        return embeddings


def _load_movie_search() -> SemanticSearch:
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(load_movies())
    return semantic_search


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
