"""Filesystem locations used by the search engine."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MOVIES_FILE = DATA_DIR / "movies.json"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"

CACHE_DIR = PROJECT_ROOT / "cache"
INDEX_CACHE_FILE = CACHE_DIR / "inverted_index.pkl"
