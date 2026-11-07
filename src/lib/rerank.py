"""Re-rank search results with an LLM."""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import cast

from sentence_transformers import CrossEncoder

from config import CROSS_ENCODER_MODEL, RERANK_WORKERS
from lib.hybrid_search import RRFResult
from lib.llm import complete

logger = logging.getLogger("search.rerank")

RERANK_METHODS = ("individual", "batch", "cross_encoder")
MAX_SCORE = 10.0
# The reply is a single number, so a few tokens are plenty.
SCORE_MAX_TOKENS = 8
# The batch reply is a short JSON list, but the free OpenRouter models are often
# reasoning models whose thinking counts toward the limit and can leave the reply empty.
BATCH_MAX_TOKENS = 3000
# Descriptions are cut to keep prompts small: hosted LLMs cap tokens per minute, and
# a batch prompt holds every result.
DESCRIPTION_CHARS = 300


def _individual_prompt(query: str, result: RRFResult) -> str:
    movie = result.movie
    return f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {movie["title"]} - {movie["description"][:DESCRIPTION_CHARS]}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""


def _parse_score(reply: str) -> float:
    """Pull the first number out of ``reply``, clamped to 0-10 (0 if there is none)."""
    match = re.search(r"\d+(?:\.\d+)?", reply)
    return min(float(match.group()), MAX_SCORE) if match else 0.0


def _rerank_individual(
    query: str, results: list[RRFResult], limit: int
) -> list[tuple[RRFResult, str]]:
    """Score each result with its own LLM call; best ``limit`` first."""

    def score(result: RRFResult) -> float:
        reply = complete(_individual_prompt(query, result), max_tokens=SCORE_MAX_TOKENS)
        return _parse_score(reply)

    with ThreadPoolExecutor(max_workers=RERANK_WORKERS) as pool:
        scores = list(pool.map(score, results))
    scored = list(zip(results, scores, strict=True))
    # sorted() is stable, so equal scores keep their RRF order.
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [(result, f"Re-rank Score: {s:.3f}/10") for result, s in scored[:limit]]


def _batch_prompt(query: str, results: list[RRFResult]) -> str:
    doc_list_str = "\n".join(
        f"{r.movie['id']}: {r.movie['title']} - "
        f"{r.movie['description'][:DESCRIPTION_CHARS]}"
        for r in results
    )
    return f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""


def _parse_ranking(reply: str) -> list[int]:
    """Parse a JSON list of movie ids, tolerating a Markdown fence around it."""
    array = re.search(r"\[.*\]", reply, re.DOTALL)
    try:
        ids = json.loads(array.group() if array else reply)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Re-ranking reply was not a JSON list: {reply!r}") from e
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):  # pyright: ignore[reportUnknownVariableType]
        raise RuntimeError(f"Re-ranking reply was not a list of ids: {reply!r}")
    return ids  # pyright: ignore[reportUnknownVariableType]


def _rerank_batch(
    query: str, results: list[RRFResult], limit: int
) -> list[tuple[RRFResult, str]]:
    """Rank all results with a single LLM call; best ``limit`` first."""
    reply = complete(_batch_prompt(query, results), max_tokens=BATCH_MAX_TOKENS)
    by_id = {r.movie["id"]: r for r in results}
    ordered: list[RRFResult] = []
    for movie_id in _parse_ranking(reply):
        result = by_id.pop(movie_id, None)  # unknown or repeated ids are skipped
        if result is not None:
            ordered.append(result)
    # Anything the LLM left out keeps its RRF order, after the ranked ones.
    ordered.extend(by_id.values())
    return [
        (result, f"Re-rank Rank: {rank}")
        for rank, result in enumerate(ordered[:limit], start=1)
    ]


@cache
def _cross_encoder() -> CrossEncoder:
    """The cross-encoder model, loaded once per process (loading is slow)."""
    return CrossEncoder(CROSS_ENCODER_MODEL)


def _rerank_cross_encoder(
    query: str, results: list[RRFResult], limit: int
) -> list[tuple[RRFResult, str]]:
    """Score every (query, movie) pair with a local cross-encoder; best first."""
    pairs = [
        [query, f"{result.movie['title']} - {result.movie['description']}"]
        for result in results
    ]
    # One score per pair; higher means more relevant (the scale is not bounded).
    scores = cast(list[float], _cross_encoder().predict(pairs).tolist())  # pyright: ignore[reportUnknownMemberType]
    scored = sorted(
        zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    logger.debug(
        "Cross-encoder scores for all %d results:\n%s",
        len(scored),
        "\n".join(
            f"  {position}. {result.movie['title']}: {score:.3f}"
            f" (rrf={result.rrf_score:.4f})"
            for position, (result, score) in enumerate(scored, start=1)
        ),
    )
    return [(result, f"Cross Encoder Score: {s:.3f}") for result, s in scored[:limit]]


def rerank(
    method: str, query: str, results: list[RRFResult], limit: int
) -> list[tuple[RRFResult, str]]:
    """Re-rank ``results`` with ``method``; each is paired with its re-rank label."""
    match method:
        case "individual":
            return _rerank_individual(query, results, limit)
        case "batch":
            return _rerank_batch(query, results, limit)
        case "cross_encoder":
            return _rerank_cross_encoder(query, results, limit)
        case _:
            raise ValueError(f"unknown re-rank method: {method!r}")
