"""Hybrid search combining BM25 keyword search with chunked semantic search."""

from dataclasses import dataclass

from config import INDEX_CACHE_FILE
from lib.dataset import Movie
from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch


@dataclass(slots=True)
class HybridResult:
    movie: Movie
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0


@dataclass(slots=True)
class RRFResult:
    movie: Movie
    bm25_rank: int | None = None  # 1-based; None if BM25 did not return the movie
    semantic_rank: int | None = None
    rrf_score: float = 0.0


def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion contribution of a 1-based ``rank``."""
    return 1 / (k + rank)


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    """Blend two normalised scores: ``alpha`` weights BM25, the rest semantic."""
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalise ``scores`` to the range [0, 1].

    If every score is equal there is no range to scale, so all become 1.0.
    """
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if low == high:
        return [1.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not INDEX_CACHE_FILE.exists():
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[Movie, float]]:
        self.idx.load()
        return self.idx.bm25_search_with_scores(query, limit)

    def weighted_search(
        self, query: str, alpha: float, limit: int = 5
    ) -> list[HybridResult]:
        """Rank movies by ``alpha * BM25 + (1 - alpha) * semantic`` (both normalised)."""
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        # Over-fetch so movies found by only one method still get both scores.
        fetch = limit * 500
        bm25_results = self._bm25_search(query, fetch)
        semantic_results = self.semantic_search.search_chunks(query, fetch)

        bm25_norm = normalize_scores([score for _, score in bm25_results])
        semantic_norm = normalize_scores([r.score for r in semantic_results])

        by_id: dict[int, HybridResult] = {}
        for (movie, _), score in zip(bm25_results, bm25_norm, strict=True):
            by_id.setdefault(movie["id"], HybridResult(movie)).bm25_score = score
        for result, score in zip(semantic_results, semantic_norm, strict=True):
            entry = by_id.setdefault(result.movie["id"], HybridResult(result.movie))
            entry.semantic_score = score

        for entry in by_id.values():
            entry.hybrid_score = hybrid_score(
                entry.bm25_score, entry.semantic_score, alpha
            )
        ranked = sorted(by_id.values(), key=lambda e: (-e.hybrid_score, e.movie["id"]))
        return ranked[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[RRFResult]:
        """Rank movies by summing ``1 / (k + rank)`` over the BM25 and semantic ranks."""
        if k < 0:
            raise ValueError(f"k must not be negative, got {k}")
        fetch = limit * 500
        bm25_results = self._bm25_search(query, fetch)
        semantic_results = self.semantic_search.search_chunks(query, fetch)

        by_id: dict[int, RRFResult] = {}
        for rank, (movie, _) in enumerate(bm25_results, start=1):
            entry = by_id.setdefault(movie["id"], RRFResult(movie))
            entry.bm25_rank = rank
            entry.rrf_score += rrf_score(rank, k)
        for rank, result in enumerate(semantic_results, start=1):
            entry = by_id.setdefault(result.movie["id"], RRFResult(result.movie))
            entry.semantic_rank = rank
            entry.rrf_score += rrf_score(rank, k)

        ranked = sorted(by_id.values(), key=lambda e: (-e.rrf_score, e.movie["id"]))
        return ranked[:limit]
