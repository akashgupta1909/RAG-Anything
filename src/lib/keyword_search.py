"""Keyword search over the movie dataset using an inverted index."""

import pickle
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import cast

from nltk.stem import PorterStemmer  # type: ignore[import-untyped]

from config import CACHE_DIR, INDEX_CACHE_FILE
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
    return [_stem(token) for token in preprocess_text(text).split() if token not in stop_words]


def tokenize_single_term(term: str) -> str:
    """Tokenize ``term`` and ensure it yields exactly one token."""
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError(f"term must produce exactly one token, got {tokens!r}")
    return tokens[0]


class InvertedIndex:
    """Maps tokens to the documents containing them, plus per-document term counts."""

    def __init__(self) -> None:
        # token -> ids of documents containing it
        self.index: defaultdict[str, set[int]] = defaultdict(set)
        # document id -> movie
        self.docmap: dict[int, Movie] = {}
        # document id -> token counts
        self.term_frequencies: dict[int, Counter[str]] = {}

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)

    def get_documents(self, token: str) -> frozenset[int]:
        """Return the ids of documents containing ``token``."""
        return frozenset(self.index.get(token, ()))

    def get_term_frequency(self, doc_id: int, token: str) -> int:
        """Return how many times ``token`` occurs in document ``doc_id``."""
        return self.term_frequencies.get(doc_id, Counter[str]())[token]

    def build(self, on_progress: ProgressCallback | None = None) -> None:
        """Index every movie in the dataset."""
        movies = load_movies()
        total = len(movies)
        for processed, movie in enumerate(movies, start=1):
            self.docmap[movie["id"]] = movie
            self._add_document(movie["id"], f"{movie['title']} {movie['description']}")
            if on_progress is not None:
                on_progress(processed, total)

    def save(self) -> None:
        """Persist the index to a single cache file, written atomically."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = INDEX_CACHE_FILE.with_suffix(".tmp")
        with tmp_file.open("wb") as file:
            pickle.dump((self.index, self.docmap, self.term_frequencies), file)
        tmp_file.replace(INDEX_CACHE_FILE)

    def load(self) -> None:
        """Load a previously saved index from the cache file."""
        try:
            with INDEX_CACHE_FILE.open("rb") as file:
                self.index, self.docmap, self.term_frequencies = pickle.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Index cache not found at {INDEX_CACHE_FILE}. Run the 'build' command first."
            ) from e


def build_index(on_progress: ProgressCallback | None = None) -> None:
    """Build the inverted index from the dataset and save it to the cache."""
    index = InvertedIndex()
    index.build(on_progress=on_progress)
    index.save()


def keyword_search(query: str, limit: int = 5) -> list[Movie]:
    """Return up to ``limit`` movies containing any token of ``query``."""
    index = InvertedIndex()
    index.load()

    results: list[Movie] = []
    seen_ids: set[int] = set()
    for token in tokenize(query):
        for doc_id in sorted(index.get_documents(token)):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            results.append(index.docmap[doc_id])
            if len(results) >= limit:
                return results
    return results


def get_term_frequency(doc_id: int, term: str) -> int:
    """Return how many times ``term`` occurs in document ``doc_id``."""
    index = InvertedIndex()
    index.load()
    return index.get_term_frequency(doc_id, tokenize_single_term(term))
