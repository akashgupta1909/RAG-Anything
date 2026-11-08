"""Text, sparse and image encoders backed by FastEmbed (ONNX Runtime, no PyTorch).

Each encoder loads its model lazily on first use, so building the application
(e.g. for ``--help`` or the API's health check) stays fast.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path

from fastembed import ImageEmbedding, SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel

from rag.utils import get_logger

logger = get_logger(__name__)


def load_model[M](factory: Callable[..., M], model_name: str, cache_dir: Path) -> M:
    """Load from the local cache without any network call; download only if missing.

    Makes start-up independent of Hugging Face Hub availability (offline
    machines, air-gapped containers) once the models have been fetched once.
    """
    try:
        return factory(model_name, cache_dir=str(cache_dir), local_files_only=True)
    except Exception:  # not cached yet (or incomplete): fall back to downloading
        logger.info("Downloading model %s", model_name)
        return factory(model_name, cache_dir=str(cache_dir))


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class DenseEncoder:
    """Dense sentence embeddings (normalised, cosine similarity)."""

    def __init__(self, model_name: str, cache_dir: Path, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size

    @cached_property
    def _model(self) -> TextEmbedding:
        logger.info("Loading dense model %s", self.model_name)
        return load_model(TextEmbedding, self.model_name, self.cache_dir)

    @cached_property
    def dimension(self) -> int:
        return len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.embed(list(texts), batch_size=self.batch_size)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        # query_embed applies the model's query instruction where it has one.
        return next(iter(self._model.query_embed(text))).tolist()


class SparseEncoder:
    """BM25 term weights; Qdrant applies IDF at query time (``Modifier.IDF``)."""

    def __init__(self, model_name: str, cache_dir: Path, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size

    @cached_property
    def _model(self) -> SparseTextEmbedding:
        logger.info("Loading sparse model %s", self.model_name)
        return load_model(SparseTextEmbedding, self.model_name, self.cache_dir)

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
            for v in self._model.embed(list(texts), batch_size=self.batch_size)
        ]

    def embed_query(self, text: str) -> SparseVector:
        v = next(iter(self._model.query_embed(text)))
        return SparseVector(indices=v.indices.tolist(), values=v.values.tolist())


class ImageEncoder:
    """CLIP: images and text are embedded into the same 512-d space."""

    def __init__(self, image_model: str, text_model: str, cache_dir: Path) -> None:
        self.image_model_name = image_model
        self.text_model_name = text_model
        self.cache_dir = cache_dir

    @cached_property
    def _image_model(self) -> ImageEmbedding:
        logger.info("Loading image model %s", self.image_model_name)
        return load_model(ImageEmbedding, self.image_model_name, self.cache_dir)

    @cached_property
    def _text_model(self) -> TextEmbedding:
        logger.info("Loading CLIP text model %s", self.text_model_name)
        return load_model(TextEmbedding, self.text_model_name, self.cache_dir)

    @cached_property
    def dimension(self) -> int:
        return len(self.embed_text("dimension probe"))

    def embed_images(self, paths: Sequence[str | Path]) -> list[list[float]]:
        return [v.tolist() for v in self._image_model.embed([str(p) for p in paths])]

    def embed_text(self, text: str) -> list[float]:
        return next(iter(self._text_model.embed([text]))).tolist()
