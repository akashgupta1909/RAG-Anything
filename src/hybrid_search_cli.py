"""Command-line interface for hybrid search."""

import argparse
import sys

from config import HYBRID_ALPHA, RRF_K
from lib.hybrid_search import (
    HybridResult,
    RRFResult,
    normalize_scores,
    rrf_search,
    weighted_search,
)
from lib.query_enhancement import ENHANCEMENT_METHODS, enhance_query
from lib.rerank import RERANK_METHODS, rerank

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
        "--rerank-method",
        type=str,
        choices=RERANK_METHODS,
        help="Re-rank the results with an LLM",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()

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
                query = args.query
                if args.enhance:
                    enhanced = enhance_query(query, args.enhance)
                    if enhanced != query:
                        print(
                            f"Enhanced query ({args.enhance}): "
                            f"'{query}' -> '{enhanced}'\n"
                        )
                    query = enhanced
                if args.rerank_method:
                    # Re-ranking can promote results from further down, so fetch more.
                    results = rrf_search(
                        query, args.k, args.limit * RERANK_FETCH_FACTOR
                    )
                    print(
                        f"Re-ranking top {len(results)} results using "
                        f"{args.rerank_method} method..."
                    )
                    reranked = rerank(args.rerank_method, query, results, args.limit)
                    print(
                        f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n"
                    )
                    _print_rrf_results(
                        [result for result, _ in reranked],
                        [label for _, label in reranked],
                    )
                else:
                    _print_rrf_results(rrf_search(query, args.k, args.limit))
            case _:
                raise AssertionError(f"unhandled command: {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
