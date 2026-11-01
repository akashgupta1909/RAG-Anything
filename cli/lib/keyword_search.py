from lib.search_util import movies_data, Movie


def keyword_search(keyword: str) -> list[Movie]:
    data = movies_data()
    matched_movies: list[Movie] = []
    for movie in data["movies"]:
        if keyword in movie["title"].lower():
            matched_movies.append(movie)
        if len(matched_movies) >= 5:
            break
    return matched_movies


def print_movie_title(movies: list[Movie]) -> None:
    for index, movie in enumerate(movies, start=1):
        print(f'{index}. {movie["title"]}')
