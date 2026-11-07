"""Command-line interface for hybrid search."""

import argparse
import logging
import sys

from config import HYBRID_ALPHA, RRF_K
from lib.hybrid_search import (
    HybridResult,
    RRFResult,
    normalize_scores,
    rrf_search,
    weighted_search,
)
from lib.llm_evaluation import evaluate_results
from lib.query_enhancement import ENHANCEMENT_METHODS, enhance_query
from lib.rerank import RERANK_METHODS, rerank

logger = logging.getLogger("search")

DESCRIPTION_PREVIEW_CHARS = 100
RERANK_FETCH_FACTOR = 5


def _format_rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _print_weighted_results(results: list[HybridResult]) -> None:
    for position, result in enumerate(results, start=1):
        print(f"{position}. {result.movie['title']}")
        print(f"  Hybrid Score: {result.hybrid_score:.3f}")
        print(f"  BM25: {result.bm25_score:.3f}, Semantic: {result.semantic_score:.3f}")
        print(f"  {result.movie['description'][:DESCRIPTION_PREVIEW_CHARS]}...")


def _print_rrf_results(
    results: list[RRFResult], rerank_labels: list[str] | None = None
) -> None:
    for position, result in enumerate(results, start=1):
        print(f"{position}. {result.movie['title']}")
        if rerank_labels is not None:
            print(f"  {rerank_labels[position - 1]}")
        print(f"  RRF Score: {result.rrf_score:.3f}")
        print(
            f"  BM25 Rank: {_format_rank(result.bm25_rank)}, "
            f"Semantic Rank: {_format_rank(result.semantic_rank)}"
        )
        print(f"  {result.movie['description'][:DESCRIPTION_PREVIEW_CHARS]}...")
        print()


def _describe(results: list[RRFResult]) -> str:
    """One line per result for the debug log: position, title, RRF score and ranks."""
    return "\n".join(
        f"  {position}. {r.movie['title']} (rrf={r.rrf_score:.4f}, "
        f"bm25={_format_rank(r.bm25_rank)}, semantic={_format_rank(r.semantic_rank)})"
        for position, r in enumerate(results, start=1)
    )


def _print_evaluation(query: str, results: list[RRFResult]) -> None:
    scores = evaluate_results(query, results)
    logger.debug("LLM relevance scores: %s", scores)
    for position, (result, score) in enumerate(
        zip(results, scores, strict=True), start=1
    ):
        print(f"{position}. {result.movie['title']}: {score}/3")


def _run_rrf_search(args: argparse.Namespace) -> None:
    query: str = args.query
    logger.debug("Original query: '%s'", query)

    if args.enhance:
        enhanced = enhance_query(query, args.enhance)
        logger.debug("Query after %s enhancement: '%s'", args.enhance, enhanced)
        if enhanced != query:
            print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
        query = enhanced
    else:
        logger.debug("Query enhancement: none")

    if not args.rerank_method:
        results = rrf_search(query, args.k, args.limit)
        logger.debug(
            "RRF results (k=%d, limit=%d):\n%s", args.k, args.limit, _describe(results)
        )
        logger.debug("Re-ranking: none")
        _print_rrf_results(results)
        if args.evaluate:
            _print_evaluation(query, results)
        return

    # Re-ranking can promote results from further down, so fetch more.
    fetch = args.limit * RERANK_FETCH_FACTOR
    results = rrf_search(query, args.k, fetch)
    logger.debug(
        "RRF results (k=%d, fetched %d):\n%s", args.k, fetch, _describe(results)
    )
    print(f"Re-ranking top {len(results)} results using {args.rerank_method} method...")
    reranked = rerank(args.rerank_method, query, results, args.limit)
    logger.debug(
        "Final results after %s re-ranking:\n%s",
        args.rerank_method,
        "\n".join(
            f"  {position}. {r.movie['title']} ({label}, rrf={r.rrf_score:.4f})"
            for position, (r, label) in enumerate(reranked, start=1)
        ),
    )
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
    final = [result for result, _ in reranked]
    _print_rrf_results(final, [label for _, label in reranked])
    if args.evaluate:
        _print_evaluation(query, final)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="*", help="Scores to normalize"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Combine BM25 and semantic search with a weight"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=HYBRID_ALPHA,
        help="Weight of BM25 vs semantic (1.0 = BM25 only, 0.0 = semantic only)",
    )
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Combine BM25 and semantic search with Reciprocal Rank Fusion",
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "-k",
        type=int,
        default=RRF_K,
        help="RRF constant: higher flattens rank differences",
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=ENHANCEMENT_METHODS,
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Rate the results from 0 to 3 with an LLM after printing them",
    )
    rrf_parser.add_argument(
        "--debug",
        action="store_true",
        help="Log each pipeline stage (query, enhancement, RRF, re-rank) to stderr",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=RERANK_METHODS,
        help="Re-rank the results with an LLM",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if getattr(args, "debug", False):
        # Only the pipeline's own logger is verbose; third-party libraries stay quiet.
        logging.basicConfig(
            level=logging.WARNING,
            format="[%(levelname)s] %(message)s",
            stream=sys.stderr,
        )
        logger.setLevel(logging.DEBUG)

    try:
        match args.command:
            case "normalize":
                for score in normalize_scores(args.scores):
                    print(f"* {score:.4f}")
            case "weighted-search":
                _print_weighted_results(
                    weighted_search(args.query, args.alpha, args.limit)
                )
            case "rrf-search":
                _run_rrf_search(args)
            case _:
                raise AssertionError(f"unhandled command: {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
