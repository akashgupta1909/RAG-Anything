"""Split long text into smaller, overlapping chunks for embedding."""

import re

SENTENCE_BOUNDARY = r"(?<=[.!?])\s+"


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    """Split ``text`` into chunks of up to ``max_chunk_size`` sentences.

    Each chunk after the first starts with the last ``overlap`` sentences of the
    previous chunk.
    """
    if max_chunk_size <= 0:
        raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= max_chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than max_chunk_size ({max_chunk_size})"
        )

    text = text.strip()
    if not text:
        return []

    sentences = re.split(SENTENCE_BOUNDARY, text)
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]  # a lone sentence without end punctuation stays whole
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s]

    chunks: list[str] = []
    step = max_chunk_size - overlap
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start : start + max_chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_chunk_size >= len(sentences):
            break  # the rest would only repeat sentences already covered
    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split ``text`` into chunks of ``chunk_size`` words.

    Every chunk after the first is prefixed with the last ``overlap`` words of the
    previous chunk, so text cut at a chunk boundary keeps some surrounding context.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")

    words = text.split()
    # A chunk can only borrow context from the previous chunk, never further back.
    context = min(overlap, chunk_size)
    return [
        " ".join(words[max(start - context, 0) : start + chunk_size])
        for start in range(0, len(words), chunk_size)
    ]
