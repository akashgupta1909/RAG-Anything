"""Multimodal search: embed images (and later text) with a CLIP model."""

from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import CLIP_MODEL
from lib.dataset import Movie, load_movies
from lib.semantic_search import Embedding

SEARCH_RESULT_COUNT = 5


class ImageSearchResult(TypedDict):
    id: int
    title: str
    description: str
    similarity: float


class MultimodalSearch:
    def __init__(
        self, documents: list[Movie] | None = None, model_name: str = CLIP_MODEL
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.documents: list[Movie] = documents or []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = cast(
            Embedding,
            self.model.encode(  # pyright: ignore[reportUnknownMemberType]
                self.texts, show_progress_bar=True
            ),
        )

    def embed_image(self, image_path: str | Path) -> Embedding:
        """Return the embedding vector for the image at ``image_path``."""
        with Image.open(image_path) as image:
            # encode() takes a batch, so wrap the single image and unwrap the result.
            embeddings = self.model.encode([image])  # pyright: ignore[reportUnknownMemberType]
        return cast(Embedding, embeddings[0])

    def search_with_image(
        self, image_path: str | Path, limit: int = SEARCH_RESULT_COUNT
    ) -> list[ImageSearchResult]:
        """Return the ``limit`` documents whose text is closest to the image."""
        image_embedding = self.embed_image(image_path)
        # Cosine similarity: dot products divided by the vectors' lengths.
        similarities = (self.text_embeddings @ image_embedding) / (
            np.linalg.norm(self.text_embeddings, axis=1)
            * np.linalg.norm(image_embedding)
        )
        # Ties are broken by document position so results are deterministic.
        top = cast(list[int], np.argsort(-similarities, kind="stable")[:limit].tolist())
        return [
            ImageSearchResult(
                id=self.documents[i]["id"],
                title=self.documents[i]["title"],
                description=self.documents[i]["description"],
                similarity=float(similarities[i]),
            )
            for i in top
        ]


def verify_image_embedding(image_path: str) -> Embedding:
    """Embed the image at ``image_path`` and print the embedding's size."""
    embedding = MultimodalSearch().embed_image(image_path)  # no documents to embed
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
    return embedding


def image_search_command(image_path: str) -> list[ImageSearchResult]:
    """Find the movies whose text best matches the image at ``image_path``."""
    return MultimodalSearch(load_movies()).search_with_image(image_path)
