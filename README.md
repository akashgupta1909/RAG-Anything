# rag-search-engine

Keyword search (inverted index with BM25 ranking) and semantic search (sentence embeddings) over a movie dataset — the retrieval layer for a RAG pipeline.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Place the dataset at `data/movies.json` (not tracked in git).

## Usage

```bash
uv run keyword-search build             # build and cache the inverted index
uv run keyword-search search "space"    # search movies, ranked by BM25
uv run keyword-search tf 42 space       # raw term frequency of a term in a document
uv run keyword-search bm25idf space     # BM25 IDF of a term
uv run keyword-search bm25tf 42 space   # BM25 TF of a term in a document (optional k1, b)

uv run semantic-search verify           # check the embedding model loads
uv run semantic-search embed "a space adventure"   # embed a piece of text
```

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pyright           # type-check (strict)
```
