"""Command-line interface for multimodal (image and text) search."""

import argparse
import sys

from lib.multimodal_search import (
    ImageSearchResult,
    image_search_command,
    verify_image_embedding,
)

DESCRIPTION_PREVIEW_CHARS = 100


def _print_image_results(results: list[ImageSearchResult]) -> None:
    for position, result in enumerate(results, start=1):
        print(f"{position}. {result['title']} (similarity: {result['similarity']:.3f})")
        print(f"   {result['description'][:DESCRIPTION_PREVIEW_CHARS]}...")
        print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Embed an image and print the embedding's size"
    )
    verify_parser.add_argument("image", type=str, help="Path to an image file")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Find movies that match an image"
    )
    image_search_parser.add_argument("image", type=str, help="Path to an image file")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    try:
        match args.command:
            case "verify_image_embedding":
                verify_image_embedding(args.image)
            case "image_search":
                _print_image_results(image_search_command(args.image))
            case _:
                raise AssertionError(f"unhandled command: {args.command}")
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
