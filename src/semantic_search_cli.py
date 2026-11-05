"""Command-line interface for semantic search."""

import argparse
import sys

from lib.chunking import chunk_text
from lib.semantic_search import (
    SearchResult,
    embed_text,
    search_movies,
    verify_embeddings,
    verify_model,
)

DESCRIPTION_PREVIEW_CHARS = 150


def _print_results(results: list[SearchResult]) -> None:
    for result in results:
        print()
        print(f"{result.movie['title']} (score {result.score:.4f})")
        print(f"{result.movie['description'][:DESCRIPTION_PREVIEW_CHARS]}...")


def _print_chunks(chunks: list[str], chunk_size: int) -> None:
    print(f"Chunking on {chunk_size}")
    for index, chunk in enumerate(chunks):
        print(f"{index}. {chunk}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="Verify the embedding model can be loaded")

    embed_parser = subparsers.add_parser(
        "embed", help="Generate an embedding for a given text"
    )
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")

    subparsers.add_parser(
        "verify_embeddings",
        help="Verify that the movie embeddings can be loaded or created",
    )

    search_parser = subparsers.add_parser(
        "search", help="Search movies by semantic search"
    )
    search_parser.add_argument(
        "query", type=str, help="Search query for semantic search"
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into overlapping chunks of words"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Number of words per chunk"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Number of words from the previous chunk to prepend to each chunk",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    try:
        match args.command:
            case "verify":
                print("Verifying embedding model...")
                model = verify_model()
                print(f"Model Loaded: '{model}'")
                print(f"Max Sequence Length: {model.get_max_seq_length()}")
            case "embed":
                embedding = embed_text(args.text)
                print(f"Text: {args.text}")
                print(f"First 3 dimensions: {embedding[:3]}")
                print(f"Dimensions: {embedding.shape[0]}")
            case "verify_embeddings":
                print("Verifying embeddings...")
                embeddings = verify_embeddings()
                num_vectors, dimensions = embeddings.shape
                print(f"Number of docs:   {num_vectors}")
                print(
                    f"Embeddings shape: {num_vectors} vectors in {dimensions} dimensions"
                )
            case "search":
                print("searching...")
                _print_results(search_movies(args.query))
            case "chunk":
                chunks = chunk_text(args.text, args.chunk_size, args.overlap)
                _print_chunks(chunks, args.chunk_size)
            case _:
                raise AssertionError(f"unhandled command: {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
