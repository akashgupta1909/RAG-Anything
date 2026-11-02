# rag-search-engine

Keyword search over a movie dataset using an inverted index — the retrieval layer for a RAG pipeline.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Place the dataset at `data/movies.json` (not tracked in git).

## Usage

```bash
uv run keyword-search build             # build and cache the inverted index
uv run keyword-search search "space"    # search movies by keyword
uv run keyword-search tf 42 space       # term frequency of a term in a document
```

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pyright           # type-check (strict)
```
