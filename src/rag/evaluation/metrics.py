"""Retrieval metrics over ranked document lists (binary relevance).

Chunks are mapped to their document URL and de-duplicated first, so a document
retrieved as three chunks counts once, at its best rank.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def normalise_url(url: str) -> str:
    return url.split("#")[0].rstrip("/").lower()


def dedupe_ranked(urls: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(normalise_url(u) for u in urls))


def hit_rate(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(u in relevant for u in ranked[:k]) else 0.0


def recall(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def precision(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return sum(u in relevant for u in top) / len(top) if top else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, url in enumerate(ranked[:k], start=1):
        if url in relevant:
            return 1.0 / rank
    return 0.0


def ndcg(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, u in enumerate(ranked[:k], start=1)
        if u in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def retrieval_scores(
    retrieved_urls: Iterable[str], relevant_urls: Iterable[str], k: int
) -> dict[str, float]:
    ranked = dedupe_ranked(retrieved_urls)
    relevant = {normalise_url(u) for u in relevant_urls}
    return {
        f"hit@{k}": hit_rate(ranked, relevant, k),
        f"recall@{k}": recall(ranked, relevant, k),
        f"precision@{k}": precision(ranked, relevant, k),
        "mrr": reciprocal_rank(ranked, relevant, k),
        f"ndcg@{k}": ndcg(ranked, relevant, k),
    }


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in 0-100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[index]
