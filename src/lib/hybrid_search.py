"""Hybrid search: fuse BM25 keyword ranking with chunked semantic ranking."""

from dataclasses import dataclass

from config import HYBRID_ALPHA, HYBRID_CANDIDATE_MULTIPLIER, RRF_K
from lib.dataset import Movie, SearchResult, load_movies
from lib.keyword_search import load_or_build_index
from lib.semantic_search import ChunkedSemanticSearch


@dataclass(frozen=True, slots=True)
class HybridResult:
    movie: Movie
    hybrid_score: float
    # Both normalised to [0, 1]; 0.0 when that method did not return the movie.
    bm25_score: float
    semantic_score: float


@dataclass(frozen=True, slots=True)
class RRFResult:
    movie: Movie
    rrf_score: float
    # 1-based; None when that method did not return the movie.
    bm25_rank: int | None
    semantic_rank: int | None


def rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion contribution of a 1-based ``rank``."""
    return 1 / (k + rank)


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = HYBRID_ALPHA
) -> float:
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


def _require_alpha(alpha: float) -> None:
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")


def _require_k(k: int) -> None:
    if k < 0:
        raise ValueError(f"k must not be negative, got {k}")


def _movies_by_id(*result_lists: list[SearchResult]) -> dict[int, Movie]:
    return {r.movie["id"]: r.movie for results in result_lists for r in results}


def _normalized_scores_by_id(results: list[SearchResult]) -> dict[int, float]:
    normalized = normalize_scores([r.score for r in results])
    return {r.movie["id"]: s for r, s in zip(results, normalized, strict=True)}


def _ranks_by_id(results: list[SearchResult]) -> dict[int, int]:
    return {r.movie["id"]: rank for rank, r in enumerate(results, start=1)}


def fuse_weighted(
    bm25: list[SearchResult],
    semantic: list[SearchResult],
    alpha: float = HYBRID_ALPHA,
    limit: int = 5,
) -> list[HybridResult]:
    """Rank the union of two result lists by their weighted, normalised scores."""
    _require_alpha(alpha)
    bm25_scores = _normalized_scores_by_id(bm25)
    semantic_scores = _normalized_scores_by_id(semantic)
    results: list[HybridResult] = []
    for movie_id, movie in _movies_by_id(bm25, semantic).items():
        bm25_score = bm25_scores.get(movie_id, 0.0)
        semantic_score = semantic_scores.get(movie_id, 0.0)
        combined = hybrid_score(bm25_score, semantic_score, alpha)
        results.append(HybridResult(movie, combined, bm25_score, semantic_score))
    # Ties are broken by movie id so results are deterministic.
    results.sort(key=lambda r: (-r.hybrid_score, r.movie["id"]))
    return results[:limit]


def fuse_rrf(
    bm25: list[SearchResult],
    semantic: list[SearchResult],
    k: int = RRF_K,
    limit: int = 5,
) -> list[RRFResult]:
    """Rank the union of two result lists by Reciprocal Rank Fusion."""
    _require_k(k)
    bm25_ranks = _ranks_by_id(bm25)
    semantic_ranks = _ranks_by_id(semantic)
    results: list[RRFResult] = []
    for movie_id, movie in _movies_by_id(bm25, semantic).items():
        bm25_rank = bm25_ranks.get(movie_id)
        semantic_rank = semantic_ranks.get(movie_id)
        score = sum(
            rrf_score(rank, k)
            for rank in (bm25_rank, semantic_rank)
            if rank is not None
        )
        results.append(RRFResult(movie, score, bm25_rank, semantic_rank))
    # Ties are broken by movie id so results are deterministic.
    results.sort(key=lambda r: (-r.rrf_score, r.movie["id"]))
    return results[:limit]


class HybridSearch:
    """Runs BM25 and chunked semantic search over the same movies and fuses them."""

    def __init__(self, documents: list[Movie]) -> None:
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.index = load_or_build_index()

    def _candidates(
        self, query: str, limit: int
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """Fetch generous candidate lists from both methods for fusion."""
        fetch = limit * HYBRID_CANDIDATE_MULTIPLIER
        return (
            self.index.bm25_search_with_scores(query, fetch),
            self.semantic_search.search_chunks(query, fetch),
        )

    def weighted_search(
        self, query: str, alpha: float = HYBRID_ALPHA, limit: int = 5
    ) -> list[HybridResult]:
        """Rank movies by ``alpha * BM25 + (1 - alpha) * semantic`` (both normalised)."""
        _require_alpha(alpha)  # fail fast, before running both searches
        return fuse_weighted(*self._candidates(query, limit), alpha=alpha, limit=limit)

    def rrf_search(self, query: str, k: int = RRF_K, limit: int = 5) -> list[RRFResult]:
        """Rank movies by summing ``1 / (k + rank)`` over the BM25 and semantic ranks."""
        _require_k(k)  # fail fast, before running both searches
        return fuse_rrf(*self._candidates(query, limit), k=k, limit=limit)


def weighted_search(
    query: str, alpha: float = HYBRID_ALPHA, limit: int = 5
) -> list[HybridResult]:
    """Weighted hybrid search over all movies."""
    _require_alpha(alpha)  # fail fast, before loading the model and the index
    return HybridSearch(load_movies()).weighted_search(query, alpha, limit)


def rrf_search(query: str, k: int = RRF_K, limit: int = 5) -> list[RRFResult]:
    """Reciprocal Rank Fusion hybrid search over all movies."""
    _require_k(k)  # fail fast, before loading the model and the index
    return HybridSearch(load_movies()).rrf_search(query, k, limit)
