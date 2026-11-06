"""The movie dataset, stop words, and the text normalisation shared by both."""

import json
import string
from functools import cache
from typing import TypedDict, cast

from config import MOVIES_FILE, STOPWORDS_FILE

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


class Movie(TypedDict):
    id: int
    title: str
    description: str


class MoviesData(TypedDict):
    movies: list[Movie]


def preprocess_text(text: str) -> str:
    """Lowercase ``text`` and strip all ASCII punctuation."""
    return text.lower().translate(_PUNCTUATION_TABLE)


@cache
def load_movies() -> list[Movie]:
    """Load the movie dataset from disk (read once, then cached)."""
    with MOVIES_FILE.open(encoding="utf-8") as file:
        data = cast(MoviesData, json.load(file))
    return data["movies"]


@cache
def load_stop_words() -> frozenset[str]:
    """Load the normalised stop-word set from disk (read once, then cached)."""
    with STOPWORDS_FILE.open(encoding="utf-8") as file:
        return frozenset(preprocess_text(line.strip()) for line in file)
