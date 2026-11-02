from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
from .search_util import MOVIES_DATA, STOP_WORDS_DATA, Movie
from .search_util import preprocess_text
from typing import Counter, cast
from config import INDEX_CACHE_FILE, DOCMAP_CACHE_FILE, TERM_FREQ_CACHE_FILE
import pickle

stemmer = PorterStemmer()


def tokenizer(text: str) -> list[str]:
    text = preprocess_text(text)
    splited_token = text.split()
    filtered_token = [token for token in splited_token if token not in STOP_WORDS_DATA]
    stemmed_token = [cast(str, stemmer.stem(token)) for token in filtered_token]  # type: ignore[reportUnknownMemberType]
    return stemmed_token


def tokenize_single_term(term: str) -> str:
    tokens = tokenizer(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    return tokens[0]


class InvertedIndex:
    def __init__(self) -> None:
        # Which token is present in which document
        self.index: dict[str, set[int]] = {}
        # Which document corresponds to which movie
        self.docmap: dict[int, Movie] = {}
        # What is the frequency of each token in each document
        self.term_frequencies: dict[int, Counter[str]] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenizer(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, token: str) -> set[int]:
        return self.index.get(token, set())

    def get_term_frequency(self, doc_id: int, token: str) -> int:
        return self.term_frequencies.get(doc_id, Counter()).get(token, 0)

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
        with open(TERM_FREQ_CACHE_FILE, "wb") as file:
            pickle.dump(self.term_frequencies, file)

    def load(self) -> None:
        try:
            with open(INDEX_CACHE_FILE, "rb") as file:
                self.index = pickle.load(file)
            with open(DOCMAP_CACHE_FILE, "rb") as file:
                self.docmap = pickle.load(file)
            with open(TERM_FREQ_CACHE_FILE, "rb") as file:
                self.term_frequencies = pickle.load(file)
        except FileNotFoundError:
            print("Cache files not found. Please build the index first.")
            raise


def build_command(show_percentage: bool = False) -> None:
    index = InvertedIndex()

    index.build(show_percentage=show_percentage)
    index.save()
    print("Inverted index and docmap have been built and saved.")


def keyword_search(keyword: str, limit: int = 5) -> list[Movie]:
    index = InvertedIndex()
    index.load()

    keyword_tokens = tokenizer(keyword)
    matched_movies: list[Movie] = []

    for token in keyword_tokens:
        doc_ids = index.get_documents(token)
        for doc_id in doc_ids:
            movie = index.docmap[doc_id]
            if movie not in matched_movies:
                matched_movies.append(movie)
                if len(matched_movies) >= limit:
                    return matched_movies

    return matched_movies


def get_term_frequency(doc_id: int, token: str) -> int:
    index = InvertedIndex()
    index.load()
    return index.get_term_frequency(doc_id, tokenize_single_term(token))


def print_movie_title(movies: list[Movie]) -> None:
    for index, movie in enumerate(movies, start=1):
        print(f'{index}. ({movie["id"]}) {movie["title"]}')


def match_keywords_in_title(title: list[str], keywords: list[str]) -> bool:
    return any(keyword in title for keyword in keywords)
