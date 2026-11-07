"""Rewrite a text query using an image, with a multimodal LLM (Groq, then OpenRouter)."""

import argparse
import base64
import mimetypes
import sys

from openai.types.chat import ChatCompletionContentPartParam

from lib.llm import complete

# Short rewritten query, with headroom for models that "think" first.
MAX_TOKENS = 1000

SYSTEM_PROMPT = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Query Rewriting CLI")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument(
        "--query", required=True, help="Text query to rewrite based on the image"
    )
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    try:
        with open(args.image, "rb") as file:
            img = file.read()

        data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"
        parts: list[ChatCompletionContentPartParam] = [
            {"type": "text", "text": SYSTEM_PROMPT.strip()},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": args.query.strip()},
        ]
        content = complete(parts, max_tokens=MAX_TOKENS)
        if not content:
            raise RuntimeError("No text in API response")
        print(f"Rewritten query: {content}")
    except (OSError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
