"""Command-line interface for keyword search."""

import argparse
import sys

from lib.keyword_search import build_index, get_term_frequency, keyword_search
from lib.search_util import Movie


def _print_movies(movies: list[Movie]) -> None:
    for position, movie in enumerate(movies, start=1):
        print(f"{position}. ({movie['id']}) {movie['title']}")


def _print_progress(processed: int, total: int) -> None:
    percentage = processed / total * 100
    print(f"Processed {percentage:.0f}% of {total} movies", end="\r", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Display term frequency for a given keyword")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to check frequency")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    try:
        match args.command:
            case "search":
                print(f"Searching for: {args.query}")
                _print_movies(keyword_search(args.query))
            case "build":
                print("Building the inverted index...")
                build_index(on_progress=_print_progress)
                print()
                print("Inverted index has been built and saved.")
            case "tf":
                frequency = get_term_frequency(args.doc_id, args.term)
                print(f"Term frequency of '{args.term}' in document {args.doc_id}: {frequency}")
            case _:
                raise AssertionError(f"unhandled command: {args.command}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
