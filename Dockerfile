# syntax=docker/dockerfile:1.7
# ---- build: resolve and install dependencies with uv -----------------------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---- runtime: slim image, non-root user ------------------------------------
FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 rag
WORKDIR /app
COPY --from=build --chown=rag:rag /app /app
COPY --chown=rag:rag main.py config.yaml ./
COPY --chown=rag:rag eval/golden_set.json ./eval/golden_set.json
RUN mkdir -p data/models data/raw logs && chown -R rag:rag data logs
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER rag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["python", "main.py", "serve"]
