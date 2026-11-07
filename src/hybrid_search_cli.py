"""Command-line interface for hybrid search."""

import argparse
import sys

from lib.dataset import load_movies
from lib.hybrid_search import HybridSearch, normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

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
        default=0.5,
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
        default=60,
        help="RRF constant: higher flattens rank differences",
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize_scores(args.scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            try:
                results = HybridSearch(load_movies()).weighted_search(
                    args.query, args.alpha, args.limit
                )
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result.movie['title']}")
                print(f"  Hybrid Score: {result.hybrid_score:.3f}")
                print(
                    f"  BM25: {result.bm25_score:.3f}, "
                    f"Semantic: {result.semantic_score:.3f}"
                )
                print(f"  {result.movie['description'][:100]}...")
        case "rrf-search":
            try:
                rrf_results = HybridSearch(load_movies()).rrf_search(
                    args.query, args.k, args.limit
                )
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            for i, rrf in enumerate(rrf_results, start=1):
                bm25_rank = "-" if rrf.bm25_rank is None else rrf.bm25_rank
                semantic_rank = "-" if rrf.semantic_rank is None else rrf.semantic_rank
                print(f"{i}. {rrf.movie['title']}")
                print(f"  RRF Score: {rrf.rrf_score:.3f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"  {rrf.movie['description'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
