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

# Hybrid search
# Default weight of BM25 in weighted search (1.0 = BM25 only, 0.0 = semantic only)
HYBRID_ALPHA = 0.5
# Reciprocal Rank Fusion constant: higher values flatten the gap between ranks
RRF_K = 60
# Each method fetches this many times ``limit`` candidates before fusing, so a
# movie found by one method usually has a score from the other one too.
HYBRID_CANDIDATE_MULTIPLIER = 500

# Query enhancement (LLM served through OpenRouter; key read from OPENROUTER_API_KEY)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ENHANCEMENT_MODEL = "openrouter/free"
