"""Split long text into smaller, overlapping chunks for embedding."""


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
