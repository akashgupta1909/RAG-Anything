"""Command-line interface for semantic search."""

import argparse
import sys

from lib.chunking import chunk_text, semantic_chunk
from lib.semantic_search import (
    SearchResult,
    embed_chunks,
    embed_text,
    search_chunked,
    search_movies,
    verify_embeddings,
    verify_model,
)

DESCRIPTION_PREVIEW_CHARS = 150
CHUNKED_PREVIEW_CHARS = 100


def _print_results(results: list[SearchResult]) -> None:
    for result in results:
        print()
        print(f"{result.movie['title']} (score {result.score:.4f})")
        print(f"{result.movie['description'][:DESCRIPTION_PREVIEW_CHARS]}...")


def _print_chunks(chunks: list[str], chunk_size: int) -> None:
    print(f"Chunking on {chunk_size}")
    for index, chunk in enumerate(chunks):
        print(f"{index}. {chunk}")


def _print_chunked_results(results: list[SearchResult]) -> None:
    for position, result in enumerate(results, start=1):
        score = round(result.score, 3)  # shown to 3 decimals, padded to 4 places
        print(f"\n{position}. {result.movie['title']} (score: {score:.4f})")
        print(f"   {result.movie['description'][:CHUNKED_PREVIEW_CHARS]}...")


def _print_semantic_chunks(text: str, chunks: list[str]) -> None:
    print(f"Semantically chunking {len(text)} characters")
    for index, chunk in enumerate(chunks, start=1):
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

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text into chunks of sentences"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences shared between consecutive chunks",
    )

    subparsers.add_parser(
        "embed_chunks", help="Generate (or load cached) embeddings for movie chunks"
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search movies by their best-matching chunk"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
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
            case "semantic_chunk":
                chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
                _print_semantic_chunks(args.text, chunks)
            case "embed_chunks":
                embeddings = embed_chunks()
                print(f"Generated {len(embeddings)} chunked embeddings")
            case "search_chunked":
                _print_chunked_results(search_chunked(args.query, args.limit))
            case _:
                raise AssertionError(f"unhandled command: {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
