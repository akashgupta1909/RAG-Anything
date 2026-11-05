"""Keyword search over the movie dataset using an inverted index ranked by BM25."""

import math
import pickle
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import cast

from nltk.stem import PorterStemmer  # type: ignore[import-untyped]

from config import BM25_B, BM25_K1, CACHE_DIR, INDEX_CACHE_FILE
from lib.search_util import (
    Movie,
    load_movies,
    load_stop_words,
    preprocess_text,
)

ProgressCallback = Callable[[int, int], None]
"""Called as ``on_progress(processed, total)`` while the index is built."""

_stemmer = PorterStemmer()


def _stem(token: str) -> str:
    return cast(str, _stemmer.stem(token))  # pyright: ignore[reportUnknownMemberType]


def tokenize(text: str) -> list[str]:
    """Normalise ``text``, drop stop words and return the stemmed tokens."""
    stop_words = load_stop_words()
    return [
        _stem(token)
        for token in preprocess_text(text).split()
        if token not in stop_words
    ]


def tokenize_single_term(term: str) -> str:
    """Tokenize ``term`` and ensure it yields exactly one token."""
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError(f"term must produce exactly one token, got {tokens!r}")
    return tokens[0]


class InvertedIndex:
    """Maps tokens to the documents containing them, plus per-document statistics."""

    def __init__(self) -> None:
        # token -> ids of documents containing it
        self.index: defaultdict[str, set[int]] = defaultdict(set)
        # document id -> movie
        self.docmap: dict[int, Movie] = {}
        # document id -> token counts
        self.term_frequencies: dict[int, Counter[str]] = {}
        # document id -> number of tokens
        self.doc_lengths: dict[int, int] = {}
        # derived from doc_lengths; refreshed after build/load, not persisted
        self._avg_doc_length = 0.0

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def _refresh_stats(self) -> None:
        num_docs = len(self.doc_lengths)
        self._avg_doc_length = (
            sum(self.doc_lengths.values()) / num_docs if num_docs else 0.0
        )

    def _require_document(self, doc_id: int) -> None:
        if doc_id not in self.docmap:
            raise ValueError(f"unknown document id: {doc_id}")

    def get_documents(self, token: str) -> frozenset[int]:
        """Return the ids of documents containing ``token``."""
        return frozenset(self.index.get(token, ()))

    def get_term_frequency(self, doc_id: int, token: str) -> int:
        """Return how many times ``token`` occurs in document ``doc_id``."""
        return self.term_frequencies.get(doc_id, Counter[str]())[token]

    def get_bm25_idf(self, token: str, smoothing: float = 0.5) -> float:
        """Return the BM25 inverse document frequency (IDF) of ``token``."""
        doc_freq = len(self.index.get(token, ()))
        total_docs = len(self.docmap)
        return math.log(
            (total_docs - doc_freq + smoothing) / (doc_freq + smoothing) + 1
        )

    def get_bm25_tf(
        self, doc_id: int, token: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        """Return the saturated, length-normalised BM25 term frequency of ``token``."""
        self._require_document(doc_id)
        tf = self.get_term_frequency(doc_id, token)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self._avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25_search(self, query: str, limit: int = 5) -> list[Movie]:
        """Return up to ``limit`` movies ranked by BM25 score for ``query``."""
        scores: defaultdict[int, float] = defaultdict(float)
        for token in set(tokenize(query)):
            idf = self.get_bm25_idf(token)
            for doc_id in self.index.get(token, ()):
                scores[doc_id] += idf * self.get_bm25_tf(doc_id, token)
        # Ties are broken by document id so results are deterministic.
        ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return [self.docmap[doc_id] for doc_id in ranked[:limit]]

    def build(self, on_progress: ProgressCallback | None = None) -> None:
        """Index every movie in the dataset."""
        movies = load_movies()
        total = len(movies)
        for processed, movie in enumerate(movies, start=1):
            self.docmap[movie["id"]] = movie
            self._add_document(movie["id"], f"{movie['title']} {movie['description']}")
            if on_progress is not None:
                on_progress(processed, total)
        self._refresh_stats()

    def save(self) -> None:
        """Persist the index to a single cache file, written atomically."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = INDEX_CACHE_FILE.with_suffix(".tmp")
        with tmp_file.open("wb") as file:
            pickle.dump(
                (self.index, self.docmap, self.term_frequencies, self.doc_lengths), file
            )
        tmp_file.replace(INDEX_CACHE_FILE)

    def load(self) -> None:
        """Load a previously saved index from the cache file."""
        try:
            with INDEX_CACHE_FILE.open("rb") as file:
                self.index, self.docmap, self.term_frequencies, self.doc_lengths = (
                    pickle.load(file)
                )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Index cache not found at {INDEX_CACHE_FILE}. Run the 'build' command first."
            ) from e
        self._refresh_stats()


def _load_index() -> InvertedIndex:
    index = InvertedIndex()
    index.load()
    return index


def build_index(on_progress: ProgressCallback | None = None) -> None:
    """Build the inverted index from the dataset and save it to the cache."""
    index = InvertedIndex()
    index.build(on_progress=on_progress)
    index.save()


def keyword_search(query: str, limit: int = 5) -> list[Movie]:
    """Return up to ``limit`` movies ranked by BM25 score for ``query``."""
    return _load_index().bm25_search(query, limit=limit)


def get_term_frequency(doc_id: int, term: str) -> int:
    """Return how many times ``term`` occurs in document ``doc_id``."""
    return _load_index().get_term_frequency(doc_id, tokenize_single_term(term))


def get_bm25_idf(term: str) -> float:
    """Return the BM25 inverse document frequency (IDF) of ``term``."""
    return _load_index().get_bm25_idf(tokenize_single_term(term))


def get_bm25_tf(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    """Return the BM25 term frequency of ``term`` in document ``doc_id``."""
    return _load_index().get_bm25_tf(doc_id, tokenize_single_term(term), k1=k1, b=b)
