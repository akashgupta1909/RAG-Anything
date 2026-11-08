"""Diagram extraction: find the images a document references and make them local.

CLIP needs the image bytes, so every reference is resolved to a file:

1. a path relative to the document (``images/lb.png``);
2. a file with the same name elsewhere in the repository (remote URLs such as
   imgur links often mirror a file shipped with the repo);
3. if the source allows it, a download into a cache folder inside the source
   (``.rag-images/``), reused on later runs.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import httpx

from rag.models import ImageRef
from rag.utils import get_logger, sha256
from rag.utils.markdown import IMAGE_TOKEN, push_heading

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
CACHE_DIR_NAME = ".rag-images"

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_HTML_IMG_RE = re.compile(r"<img\b[^>]*?\bsrc=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
_HTML_ALT_RE = re.compile(r"\balt=[\"']([^\"']*)[\"']", re.IGNORECASE)
_USELESS_ALT = {"", "imgur", "image", "img", "diagram", "screenshot"}


class ImageResolver:
    """Resolves image references of one source to local files."""

    def __init__(self, root: Path, *, download: bool = False) -> None:
        self.root = root.resolve()
        self.download = download
        self.cache_dir = self.root / CACHE_DIR_NAME
        self._by_name: dict[str, Path] | None = None
        self._client: httpx.Client | None = None

    def resolve(self, src: str, doc_dir: Path) -> tuple[Path, bool] | None:
        """Return ``(path, downloaded)`` for ``src``, or ``None`` if unavailable."""
        name = PurePosixPath(src.split("?")[0].split("#")[0]).name
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            return None
        remote = "://" in src
        if not remote:
            candidate = (doc_dir / src).resolve()
            if candidate.is_file() and candidate.is_relative_to(self.root):
                return candidate, False
        if (local := self._index().get(name)) is not None:
            return local, False
        if remote and self.download:
            path = self._fetch(src, suffix)
            return (path, True) if path else None
        return None

    def _index(self) -> dict[str, Path]:
        if self._by_name is None:
            self._by_name = {
                p.name: p
                for p in self.root.rglob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
                and ".git" not in p.parts
                and CACHE_DIR_NAME not in p.parts
            }
        return self._by_name

    def _fetch(self, url: str, suffix: str) -> Path | None:
        path = self.cache_dir / f"{sha256(url)[:24]}{suffix}"
        if path.exists():
            return path
        if self._client is None:
            self._client = httpx.Client(
                timeout=20,
                follow_redirects=True,
                headers={"User-Agent": "rag-search-engine"},
            )
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.debug("Image download failed for %s: %s", url, e)
            return None
        if not response.headers.get("content-type", "").startswith("image/"):
            return None
        if len(response.content) > MAX_IMAGE_BYTES:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return path

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def extract_images(
    markdown: str,
    *,
    doc_dir: Path,
    resolver: ImageResolver,
    asset_base: str,
    fallback_caption: str = "",
) -> tuple[str, list[ImageRef]]:
    """Replace image references with ``[[image:<id>]]`` tokens and collect them.

    Images that cannot be resolved to a local file are dropped from the text.
    A diagram's caption is its alt text, else the enclosing heading, else
    ``fallback_caption`` (the document title).
    """
    images: dict[str, ImageRef] = {}
    stack: list[tuple[int, str]] = []
    in_fence = False
    out_lines: list[str] = []

    def replace(src: str, alt: str) -> str:
        resolved = resolver.resolve(src, doc_dir)
        if resolved is None:
            return ""
        path, downloaded = resolved
        image_id = sha256(path.read_bytes())[:16]
        if image_id not in images:
            headings = [text for _, text in stack]
            if downloaded or not asset_base:
                url = src
            else:
                url = asset_base + path.relative_to(resolver.root).as_posix()
            caption = alt.strip() if alt.strip().lower() not in _USELESS_ALT else ""
            images[image_id] = ImageRef(
                image_id=image_id,
                local_path=str(path),
                url=url,
                caption=caption
                or (headings[-1] if headings else fallback_caption or path.stem),
                heading_path=headings,
            )
        return IMAGE_TOKEN.format(image_id)

    for line in markdown.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            if heading := _HEADING_RE.match(line):
                stack = push_heading(
                    stack, len(heading.group(1)), heading.group(2).strip()
                )
            line = _MD_IMAGE_RE.sub(lambda m: replace(m.group(2), m.group(1)), line)
            line = _HTML_IMG_RE.sub(
                lambda m: replace(
                    m.group(1),
                    alt.group(1) if (alt := _HTML_ALT_RE.search(m.group(0))) else "",
                ),
                line,
            )
        out_lines.append(line)
    return "\n".join(out_lines), list(images.values())
