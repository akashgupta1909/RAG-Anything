from config import MOVIES_FILE, STOPWORDS_FILE
import json
from typing import TypedDict
import string
from functools import cache


class Movie(TypedDict):
    id: int
    title: str
    description: str


class MoviesData(TypedDict):
    movies: list[Movie]


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


@cache
def movies_data() -> MoviesData:
    with open(MOVIES_FILE) as file:
        data = json.load(file)
    return data


@cache
def stop_words_data() -> list[str]:
    with open(STOPWORDS_FILE) as file:
        stop_words = [line.strip() for line in file]
        stop_words = [preprocess_text(word) for word in stop_words]
    return stop_words


MOVIES_DATA = movies_data()
STOP_WORDS_DATA = stop_words_data()
