from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
MOVIES_FILE = DATA_DIR / "movies.json"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"
INDEX_CACHE_FILE = CACHE_DIR / "index.pkl"
DOCMAP_CACHE_FILE = CACHE_DIR / "docmap.pkl"
