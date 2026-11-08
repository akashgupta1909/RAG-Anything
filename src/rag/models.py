"""Core data models shared by every layer of the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageRef(BaseModel):
    """A diagram referenced from a document."""

    image_id: str
    local_path: str  # where the file lives on disk (for CLIP embedding)
    url: str  # public URL (for display)
    caption: str  # alt text or, failing that, the enclosing section heading
    heading_path: list[str] = Field(default_factory=list)


class Document(BaseModel):
    """One source file, cleaned and converted to Markdown."""

    doc_id: str  # "<domain>/<source>/<relative path>", unique and stable
    domain: str
    source: str
    title: str
    url: str
    text: str  # Markdown
    images: list[ImageRef] = Field(default_factory=list)


class Chunk(BaseModel):
    """A retrievable piece of a document."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    domain: str
    title: str
    url: str
    heading_path: list[str]
    text: str
    image_ids: list[str] = Field(default_factory=list)

    @property
    def breadcrumb(self) -> str:
        return " > ".join([self.title, *self.heading_path])

    @property
    def embedding_text(self) -> str:
        """Text sent to the encoders: the chunk prefixed with where it comes from.

        This "contextual chunk header" lets a chunk such as "Use ``--force``..."
        still match queries that name the page or section it belongs to.
        """
        return f"{self.breadcrumb}\n\n{self.text}"


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its scores."""

    chunk_id: str
    doc_id: str
    domain: str
    title: str
    url: str
    heading_path: list[str]
    text: str
    image_ids: list[str] = Field(default_factory=list)
    score: float  # fusion (or single-method) score from the vector DB
    rerank_score: float | None = None
    query: str | None = None  # the (sub-)query that retrieved it

    @property
    def breadcrumb(self) -> str:
        return " > ".join([self.title, *self.heading_path])


class Diagram(BaseModel):
    """An indexed image, as returned to clients."""

    image_id: str
    url: str
    caption: str
    domain: str
    doc_id: str
    doc_url: str
    heading_path: list[str] = Field(default_factory=list)
    score: float | None = None
