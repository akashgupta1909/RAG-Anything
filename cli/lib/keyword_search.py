from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
from .search_util import MOVIES_DATA, STOP_WORDS_DATA, Movie
from .search_util import preprocess_text
from typing import cast
from config import INDEX_CACHE_FILE, DOCMAP_CACHE_FILE
import pickle
from functools import cache

stemmer = PorterStemmer()


def tokenizer(text: str) -> list[str]:
    text = preprocess_text(text)
    splited_token = text.split()
    filtered_token = [token for token in splited_token if token not in STOP_WORDS_DATA]
    stemmed_token = [cast(str, stemmer.stem(token)) for token in filtered_token]  # type: ignore[reportUnknownMemberType]
    return stemmed_token


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenizer(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, token: str) -> set[int]:
        return self.index.get(token, set())

    def build(self, show_percentage: bool = False) -> None:
        for movie in MOVIES_DATA["movies"]:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            if show_percentage:
                percentage = (movie["id"] / len(MOVIES_DATA["movies"])) * 100
                print(
                    f"Processed {percentage:.0f}% of {len(MOVIES_DATA['movies'])} movies",
                    end="\r",
                    flush=True,
                )

    def save(self) -> None:
        with open(INDEX_CACHE_FILE, "wb") as file:
            pickle.dump(self.index, file)
        with open(DOCMAP_CACHE_FILE, "wb") as file:
            pickle.dump(self.docmap, file)


@cache
def build_command(show_percentage: bool = False) -> None:
    index = InvertedIndex()
    index.build(show_percentage=show_percentage)
    index.save()
    print("Inverted index and docmap have been built and saved.")


def keyword_search(keyword: str, limit: int = 5) -> list[Movie]:
    keyword_tokens = tokenizer(keyword)
    matched_movies: list[Movie] = []
    for movie in MOVIES_DATA["movies"]:
        title_tokens = tokenizer(movie["title"])
        if match_keywords_in_title(title_tokens, keyword_tokens):
            matched_movies.append(movie)
        if len(matched_movies) >= limit:
            break
    return matched_movies


def print_movie_title(movies: list[Movie]) -> None:
    for index, movie in enumerate(movies, start=1):
        print(f'{index}. {movie["title"]}')


def match_keywords_in_title(title: list[str], keywords: list[str]) -> bool:
    return any(keyword in title for keyword in keywords)
