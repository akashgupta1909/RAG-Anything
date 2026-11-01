import argparse
from lib.keyword_search import keyword_search, print_movie_title


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = keyword_search(args.query)
            print_movie_title(movies)
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
