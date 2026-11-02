import argparse
from lib.keyword_search import (
    keyword_search,
    print_movie_title,
    build_command,
    get_term_frequency,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    term_freq_parser = subparsers.add_parser(
        "tf", help="Display term frequency for a given keyword"
    )
    term_freq_parser.add_argument("doc_id", type=int, help="Document ID")
    term_freq_parser.add_argument("term", type=str, help="Term to check frequency")

    args = parser.parse_args()

    try:
        match args.command:
            case "search":
                movies = keyword_search(args.query)
                print(f"Searching for: {args.query}")
                print_movie_title(movies)
            case "build":
                print("Building the inverted index...")
                build_command(show_percentage=True)
            case "tf":
                frequency = get_term_frequency(args.doc_id, args.term)
                print(
                    f"Term frequency of '{args.term}' in document {args.doc_id}: {frequency}"
                )
            case _:
                parser.print_help()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
