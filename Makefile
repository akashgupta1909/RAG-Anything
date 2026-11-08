.PHONY: install ingest serve test lint format typecheck eval check qdrant

install:     ## Install dependencies (incl. dev tools)
	uv sync

ingest:      ## Fetch, chunk, embed and index all domains
	uv run python main.py ingest

serve:       ## Run the API on http://localhost:8000/docs
	uv run python main.py serve --reload

qdrant:      ## Start a Qdrant server (then set QDRANT_URL=http://localhost:6333)
	docker compose up -d qdrant

eval:        ## Retrieval ablation on the golden set
	uv run python main.py eval --ablation

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyright

check: lint typecheck test  ## Everything CI would run
