from config import MOVIES_FILE
import json
from typing import TypedDict


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
