"""Turn raw source files (Markdown, MDX, HTML) into clean Markdown ``Document``s.

Pipeline per file: front matter -> (HTML -> Markdown) -> markup cleaners
(see ``cleaners.py``) -> diagram extraction (see ``images.py``) -> whitespace
and link normalisation.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from bs4 import BeautifulSoup, Tag
from markdownify import MarkdownConverter

from rag.config import SourceConfig
from rag.ingestion.cleaners import clean_markup, outside_fences, split_front_matter
from rag.ingestion.images import ImageResolver, extract_images
from rag.models import Document, ImageRef

_HTML_EXTENSIONS = {".html", ".htm"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Presentation-only tags; their text content is kept.
_INLINE_HTML_RE = re.compile(
    r"</?(?:p|br|i|b|em|strong|a|div|span|sup|sub|center|small|font|figure|figcaption|details|summary|ins|hr|kbd)\b[^>]*>",
    re.IGNORECASE,
)
# [text](target "title") -> text: link targets are noise for embeddings and BM25.
_MD_LINK_RE = re.compile(
    r"(?<!!)\[([^\]\n]+)\]\((?:[^()\s]|\([^()]*\))*(?:\s+\"[^\"]*\")?\)"
)
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_MIN_DOCUMENT_CHARS = 50


# --- HTML (Sphinx Python docs, Go website) ------------------------------------
class _DocsConverter(MarkdownConverter):
    """markdownify with fenced code blocks that keep their language."""

    def convert_pre(self, el: Tag, text: str, parent_tags: set[str]) -> str:  # type: ignore[override]
        language = ""
        for parent in el.parents:
            classes = parent.get("class") or []
            found = [
                c.removeprefix("highlight-")
                for c in classes
                if c.startswith("highlight-")
            ]
            if found:
                language = (
                    "python"
                    if found[0] in {"python3", "pycon", "default"}
                    else found[0]
                )
                break
        code = el.get_text().strip("\n")
        return f"\n\n```{language}\n{code}\n```\n\n"


def html_to_markdown(html: str) -> tuple[str, str]:
    """Return ``(title, markdown)`` for the main content of an HTML page or fragment."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one('div[role="main"]') or soup.body or soup
    for selector in ("a.headerlink", "script", "style", "nav", "div.related", "footer"):
        for element in main.select(selector):
            element.decompose()
    h1 = main.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    markdown = _DocsConverter(heading_style="ATX", bullets="-").convert_soup(main)
    return title, markdown


# --- Shared post-processing ------------------------------------------------------
def _first_heading(markdown: str) -> str:
    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return ""


def _tidy(text: str) -> str:
    return _MD_LINK_RE.sub(r"\1", _INLINE_HTML_RE.sub("", text))


def normalise(markdown: str) -> str:
    """Strip comments, presentational HTML and link targets; tidy whitespace."""
    markdown = outside_fences(_HTML_COMMENT_RE.sub("", markdown), _tidy)
    markdown = "\n".join(line.rstrip() for line in markdown.splitlines())
    return _BLANK_LINES_RE.sub("\n\n", markdown).strip()


class DocumentLoader:
    """Loads every file of one source as a ``Document``."""

    def __init__(self, domain: str, source: SourceConfig, root: Path) -> None:
        self.domain = domain
        self.source = source
        self.root = root
        self._resolver = ImageResolver(root, download=source.download_images)

    def _format(self, path: Path) -> str:
        if self.source.format != "auto":
            return self.source.format
        return "html" if path.suffix.lower() in _HTML_EXTENSIONS else "markdown"

    def load(self, path: Path, relative: str) -> Document | None:
        """Return the document, or ``None`` if the file is a draft or has no content."""
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = split_front_matter(raw)
        if meta.get("draft") is True:
            return None

        title = str(meta.get("title") or meta.get("linktitle") or "")
        if self._format(path) == "html":
            html_title, body = html_to_markdown(body)
            title = title or html_title
        markdown = clean_markup(body)

        images: list[ImageRef] = []
        if self.source.images:
            markdown, images = extract_images(
                markdown,
                doc_dir=path.parent,
                resolver=self._resolver,
                asset_base=self.source.asset_base,
                fallback_caption=title,
            )
        markdown = normalise(markdown)
        if len(markdown) < _MIN_DOCUMENT_CHARS:
            return None
        title = title or _first_heading(markdown) or PurePosixPath(relative).stem
        return Document(
            doc_id=f"{self.domain}/{self.source.name}/{relative}",
            domain=self.domain,
            source=self.source.name,
            title=title.strip(),
            url=self.source.url.build(relative, meta),
            text=markdown,
            images=images,
        )

    def close(self) -> None:
        self._resolver.close()
