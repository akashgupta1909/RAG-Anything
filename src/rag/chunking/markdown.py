"""Structure-aware Markdown chunking.

Rather than cutting every N characters, the chunker follows the document's own
structure:

1. Split the document into sections at headings, tracking the heading path
   (e.g. ``["Coroutines and tasks", "Creating Tasks"]``).
2. Split each section into blocks: paragraphs, lists, tables and fenced code
   blocks. Code blocks are never cut in half unless a single one is too large.
3. Pack consecutive blocks into chunks of at most ``max_chars``.
4. Merge chunks smaller than ``min_chars`` into a neighbour from the same
   document, so short sections such as "Before you begin" keep enough context.

Every chunk keeps its heading path; ``Chunk.embedding_text`` prefixes it to the
text before embedding (a "contextual chunk header").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.models import Chunk, Document
from rag.utils import stable_uuid
from rag.utils.markdown import IMAGE_TOKEN_RE, push_heading

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*_(\[])")


@dataclass
class _Section:
    heading_path: list[str]
    blocks: list[str] = field(default_factory=list)


@dataclass
class _Piece:
    heading_path: list[str]
    text: str


def _split_sections(markdown: str) -> list[_Section]:
    """Split Markdown into sections at headings, keeping code fences intact."""
    sections = [_Section(heading_path=[])]
    stack: list[tuple[int, str]] = []
    block: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(block).strip()
        if text:
            sections[-1].blocks.append(text)
        block.clear()

    for line in markdown.splitlines():
        if _FENCE_RE.match(line):
            if not in_fence:
                flush()  # a code block starts a new block
            block.append(line)
            in_fence = not in_fence
            if not in_fence:
                flush()
            continue
        if in_fence:
            block.append(line)
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            stack = push_heading(stack, len(heading.group(1)), heading.group(2).strip())
            # The heading line stays in the text so merged chunks keep their structure.
            sections.append(
                _Section(heading_path=[t for _, t in stack], blocks=[line.strip()])
            )
        elif not line.strip():
            flush()
        else:
            block.append(line)
    flush()
    return [s for s in sections if s.blocks]


def _split_oversized(block: str, max_chars: int) -> list[str]:
    """Split a block longer than ``max_chars``: code by lines, prose by sentences."""
    is_code = bool(_FENCE_RE.match(block))
    if is_code:
        lines = block.splitlines()
        fence_open, fence_close = lines[0], "```"
        body = (
            lines[1:-1] if len(lines) > 1 and _FENCE_RE.match(lines[-1]) else lines[1:]
        )
        budget = max_chars - len(fence_open) - len(fence_close) - 2
        parts: list[str] = []
        current: list[str] = []
        size = 0
        for line in body:
            line = line[:budget]  # pathological single lines (minified code, data)
            if current and size + len(line) + 1 > budget:
                parts.append("\n".join([fence_open, *current, fence_close]))
                current, size = [], 0
            current.append(line)
            size += len(line) + 1
        if current:
            parts.append("\n".join([fence_open, *current, fence_close]))
        return parts

    units = _SENTENCE_END_RE.split(block)
    parts = []
    current_text = ""
    for unit in units:
        while len(unit) > max_chars:  # a "sentence" with no breaks: hard cut
            if current_text:
                parts.append(current_text)
                current_text = ""
            parts.append(unit[:max_chars])
            unit = unit[max_chars:]
        candidate = f"{current_text} {unit}".strip()
        if len(candidate) > max_chars and current_text:
            parts.append(current_text)
            current_text = unit
        else:
            current_text = candidate
    if current_text:
        parts.append(current_text)
    return parts


def _pack(section: _Section, max_chars: int) -> list[_Piece]:
    """Greedily pack a section's blocks into pieces of at most ``max_chars``."""
    pieces: list[_Piece] = []
    current: list[str] = []
    size = 0
    for block in section.blocks:
        for part in (
            [block] if len(block) <= max_chars else _split_oversized(block, max_chars)
        ):
            if current and size + len(part) + 2 > max_chars:
                pieces.append(_Piece(section.heading_path, "\n\n".join(current)))
                current, size = [], 0
            current.append(part)
            size += len(part) + 2
    if current:
        pieces.append(_Piece(section.heading_path, "\n\n".join(current)))
    return pieces


def _merge_small(pieces: list[_Piece], min_chars: int, max_chars: int) -> list[_Piece]:
    """Merge pieces under ``min_chars`` into the following (or previous) piece."""
    merged: list[_Piece] = []
    pending: _Piece | None = None
    for piece in pieces:
        if pending is not None:
            combined = f"{pending.text}\n\n{piece.text}"
            if len(combined) <= max_chars:
                # Keep the deepest heading path the two pieces have in common.
                common: list[str] = []
                for a, b in zip(pending.heading_path, piece.heading_path, strict=False):
                    if a != b:
                        break
                    common.append(a)
                piece = _Piece(common, combined)
            else:
                merged.append(pending)
            pending = None
        if len(piece.text) < min_chars:
            pending = piece
        else:
            merged.append(piece)
    if pending is not None:
        if merged and len(merged[-1].text) + len(pending.text) + 2 <= max_chars:
            last = merged.pop()
            merged.append(_Piece(last.heading_path, f"{last.text}\n\n{pending.text}"))
        else:
            merged.append(pending)
    return merged


class MarkdownChunker:
    """Splits ``Document``s into ``Chunk``s along their Markdown structure."""

    def __init__(self, max_chars: int = 1500, min_chars: int = 300) -> None:
        if not 0 <= min_chars < max_chars:
            raise ValueError("require 0 <= min_chars < max_chars")
        self.max_chars = max_chars
        self.min_chars = min_chars

    @property
    def signature(self) -> str:
        """Identifies the chunking settings (part of each document's content hash)."""
        return f"markdown-v1:max={self.max_chars}:min={self.min_chars}"

    def split_text(self, markdown: str) -> list[tuple[list[str], str]]:
        """Return ``(heading_path, text)`` pairs for ``markdown``."""
        pieces: list[_Piece] = []
        for section in _split_sections(markdown):
            pieces.extend(_pack(section, self.max_chars))
        merged = _merge_small(pieces, self.min_chars, self.max_chars)
        return [(p.heading_path, p.text) for p in merged if p.text.strip()]

    def chunk(self, document: Document) -> list[Chunk]:
        captions = {img.image_id: img.caption for img in document.images}
        chunks: list[Chunk] = []
        for heading_path, text in self.split_text(document.text):
            image_ids = list(dict.fromkeys(IMAGE_TOKEN_RE.findall(text)))
            # Show diagrams as readable placeholders in the chunk text.
            text = IMAGE_TOKEN_RE.sub(
                lambda m: f"[Diagram: {captions.get(m.group(1), 'figure')}]", text
            )
            # The page title usually repeats as the first heading; drop it from the path.
            if heading_path and heading_path[0] == document.title:
                heading_path = heading_path[1:]
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=stable_uuid(f"{document.doc_id}#{index}"),
                    doc_id=document.doc_id,
                    chunk_index=index,
                    domain=document.domain,
                    title=document.title,
                    url=document.url,
                    heading_path=heading_path,
                    text=text,
                    image_ids=image_ids,
                )
            )
        return chunks
