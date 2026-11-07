"""Rate search results with an LLM (a stand-in for human relevance judgements)."""

import json
import re

from lib.hybrid_search import RRFResult
from lib.llm import complete

MIN_SCORE = 0
MAX_SCORE = 3
# Descriptions are cut to keep the prompt small (hosted LLMs cap tokens per minute).
DESCRIPTION_CHARS = 300
# The reply is a short JSON list, with headroom for models that "think" first.
MAX_TOKENS = 1000


def _prompt(query: str, results: list[RRFResult]) -> str:
    formatted_results = [
        f"{position}. {r.movie['title']} - {r.movie['description'][:DESCRIPTION_CHARS]}"
        for position, r in enumerate(results, start=1)
    ]
    return f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""


def _parse_scores(reply: str, expected: int) -> list[int]:
    """Parse a JSON list of ``expected`` scores, tolerating a Markdown fence."""
    array = re.search(r"\[.*\]", reply, re.DOTALL)
    try:
        scores: object = json.loads(array.group() if array else reply)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Evaluation reply was not a JSON list: {reply!r}") from e
    if (
        not isinstance(scores, list)
        or len(scores) != expected  # pyright: ignore[reportUnknownArgumentType]
        or not all(
            isinstance(s, int) and MIN_SCORE <= s <= MAX_SCORE
            for s in scores  # pyright: ignore[reportUnknownVariableType]
        )
    ):
        raise RuntimeError(
            f"Evaluation reply must be {expected} scores from "
            f"{MIN_SCORE} to {MAX_SCORE}: {reply!r}"
        )
    return scores  # pyright: ignore[reportUnknownVariableType]


def evaluate_results(query: str, results: list[RRFResult]) -> list[int]:
    """Return a 0-3 relevance score for each result, in the same order."""
    if not results:
        return []
    reply = complete(_prompt(query, results), max_tokens=MAX_TOKENS)
    return _parse_scores(reply, len(results))
