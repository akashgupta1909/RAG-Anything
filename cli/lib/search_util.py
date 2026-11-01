from config import MOVIES_FILE
import json
from typing import TypedDict
import string


class Movie(TypedDict):
    id: int
    title: str
    description: str


class MoviesData(TypedDict):
    movies: list[Movie]


def movies_data() -> MoviesData:
    with open(MOVIES_FILE) as file:
        data = json.load(file)
    return data


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenizer(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    return tokens


def match_keywords_in_title(title: list[str], keywords: list[str]) -> bool:
    return any(keyword in title for keyword in keywords)
