"""Paths and tuning parameters used by the search engine."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LLMProvider:
    name: str
    base_url: str
    api_key_env: str  # environment variable holding the API key
    model: str


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

# LLM providers for query enhancement and re-ranking, tried in this order: a call
# falls through to the next one when a provider has no key or its request fails
# (rate limit, outage, ...). All are OpenAI-compatible APIs.
LLM_PROVIDERS = (
    LLMProvider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model="qwen/qwen3.8-27b",
    ),
    LLMProvider(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model="openrouter/free",
    ),
)
# LLM calls re-ranking runs at once
RERANK_WORKERS = 4

# Local model that re-ranks (query, movie) pairs by relevance
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
