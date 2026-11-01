from .search_util import Movie, match_keywords_in_title, movies_data, tokenizer


def keyword_search(keyword: str, limit: int = 5) -> list[Movie]:
    keyword_tokens = tokenizer(keyword)
    data = movies_data()
    matched_movies: list[Movie] = []
    for movie in data["movies"]:
        title_tokens = tokenizer(movie["title"])
        if match_keywords_in_title(title_tokens, keyword_tokens):
            matched_movies.append(movie)
        if len(matched_movies) >= limit:
            break
    return matched_movies


def print_movie_title(movies: list[Movie]) -> None:
    for index, movie in enumerate(movies, start=1):
        print(f'{index}. {movie["title"]}')
