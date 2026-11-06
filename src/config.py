"""Paths and tuning parameters used by the search engine."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MOVIES_FILE = DATA_DIR / "movies.json"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"

CACHE_DIR = PROJECT_ROOT / "cache"
INDEX_CACHE_FILE = CACHE_DIR / "inverted_index.pkl"
MOVIE_EMBEDDINGS_FILE = CACHE_DIR / "movie_embeddings.npz"
CHUNK_EMBEDDINGS_FILE = CACHE_DIR / "chunk_embeddings.npz"

# BM25 ranking parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Sentence-transformers model used for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Sentence chunking used for chunked semantic search (sizes are in sentences)
CHUNK_SIZE = 4
CHUNK_OVERLAP = 1
