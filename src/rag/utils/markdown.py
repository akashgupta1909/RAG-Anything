"""Markdown helpers shared by ingestion and chunking.

Kept in ``utils`` so that neither package imports the other (``chunking``
depends on these helpers, and ``ingestion`` depends on ``chunking``).
"""

import re

# Diagrams are replaced by ``[[image:<id>]]`` tokens during ingestion and
# turned into readable "[Diagram: caption]" placeholders by the chunker.
IMAGE_TOKEN = "[[image:{}]]"
IMAGE_TOKEN_RE = re.compile(r"\[\[image:([0-9a-f]{16})\]\]")


def push_heading(
    stack: list[tuple[int, str]], level: int, text: str
) -> list[tuple[int, str]]:
    """Return the heading stack after a heading of ``level``: deeper or equal
    headings are closed first, so skipped levels (no H1, H2 -> H4) work."""
    return [*[(lvl, t) for lvl, t in stack if lvl < level], (level, text)]
