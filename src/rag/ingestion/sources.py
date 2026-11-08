"""Fetch raw source files (git repositories, HTTP archives, local folders) to disk."""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
import zipfile
from collections.abc import Iterator
from pathlib import Path

import httpx

from rag.config import SourceConfig
from rag.utils import get_logger, sha256

logger = get_logger(__name__)


class SourceError(RuntimeError):
    """A source could not be fetched."""


def fetch_source(source: SourceConfig, raw_dir: Path, *, refresh: bool = False) -> Path:
    """Make ``source`` available locally and return its root directory.

    Downloads are cached under ``raw_dir``; ``refresh`` re-fetches them.
    """
    root = raw_dir / source.name
    match source.kind:
        case "local":
            path = Path(source.location).expanduser().resolve()
            if not path.is_dir():
                raise SourceError(
                    f"local source {source.name!r}: {path} is not a directory"
                )
            return path
        case "git":
            _fetch_git(source, root, refresh=refresh)
        case "http_archive":
            _fetch_archive(source, root, refresh=refresh)
    return root


def iter_source_files(source: SourceConfig, root: Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(absolute path, path relative to root)`` for files matching the globs."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if _matches(relative, source.include) and not _matches(
            relative, source.exclude
        ):
            yield path, relative


def _matches(path: str, patterns: list[str]) -> bool:
    # fnmatch's "*" also matches "/", so "a/**/*.md" and "a/*.md" both work as
    # recursive patterns; also try without "**/" so files directly in "a/" match.
    return any(
        fnmatch.fnmatch(path, pattern)
        or fnmatch.fnmatch(path, pattern.replace("**/", ""))
        for pattern in patterns
    )


def _run_git(*args: str, cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )
    except FileNotFoundError as e:
        raise SourceError("git is required to fetch git sources") from e
    except subprocess.CalledProcessError as e:
        raise SourceError(f"git {' '.join(args)} failed: {e.stderr.strip()}") from e


def _fetch_git(source: SourceConfig, root: Path, *, refresh: bool) -> None:
    """Shallow (and, if configured, sparse) clone; ``refresh`` pulls the latest."""
    if (root / ".git").exists():
        if refresh:
            logger.info("Updating %s", source.location)
            _run_git("pull", "--ff-only", "--depth", "1", cwd=root)
        return
    logger.info("Cloning %s", source.location)
    root.parent.mkdir(parents=True, exist_ok=True)
    args = ["clone", "--depth", "1", "--filter=blob:none"]
    if source.ref:
        args += ["--branch", source.ref]
    if source.sparse_paths:
        args.append("--sparse")
    _run_git(*args, source.location, str(root))
    if source.sparse_paths:
        _run_git("sparse-checkout", "set", *source.sparse_paths, cwd=root)


def _fetch_archive(source: SourceConfig, root: Path, *, refresh: bool) -> None:
    """Download and extract a zip archive; a marker file records which URL it holds."""
    marker = root / ".source"
    if (
        marker.exists()
        and marker.read_text() == sha256(source.location)
        and not refresh
    ):
        return
    logger.info("Downloading %s", source.location)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    archive = root.with_suffix(".zip.part")
    try:
        with (
            httpx.stream(
                "GET", source.location, follow_redirects=True, timeout=120
            ) as resp,
            archive.open("wb") as file,
        ):
            resp.raise_for_status()
            for block in resp.iter_bytes():
                file.write(block)
        with zipfile.ZipFile(archive) as zf:
            _safe_extract(zf, root)
    except (httpx.HTTPError, zipfile.BadZipFile) as e:
        shutil.rmtree(root, ignore_errors=True)
        raise SourceError(f"could not fetch {source.location}: {e}") from e
    finally:
        archive.unlink(missing_ok=True)
    marker.write_text(sha256(source.location))


def _safe_extract(zf: zipfile.ZipFile, target: Path) -> None:
    """Extract ``zf`` into ``target``, refusing entries that escape it (zip-slip)."""
    target = target.resolve()
    for member in zf.infolist():
        destination = (target / member.filename).resolve()
        if not destination.is_relative_to(target):
            raise SourceError(f"unsafe path in archive: {member.filename}")
    zf.extractall(target)
