"""FastAPI application factory.

uv run python main.py serve        # or: uvicorn rag.api.app:create_app --factory
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag import __version__
from rag.api.routes import router
from rag.config import Settings, get_settings
from rag.container import Container
from rag.llm import LLMError
from rag.utils import get_logger, setup_logging
from rag.utils.logging import request_id_var
from rag.vectordb import VectorStoreError

logger = get_logger("rag.api")


def create_app(
    settings: Settings | None = None,
    container: Container | None = None,
    *,
    warm_up: bool = True,
) -> FastAPI:
    """Build the app. Tests pass a ``container`` with fake components."""
    settings = settings or (container.settings if container else get_settings())
    setup_logging(settings.app.log_level, settings.app.log_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or Container(settings)
        if warm_up:
            logger.info("Loading models...")
            app.state.container.warm_up()
        logger.info("API ready")
        yield
        app.state.container.close()

    app = FastAPI(
        title="RAG Search Engine",
        version=__version__,
        description="Agentic RAG over programming-language, framework, Kubernetes and system design documentation.",
        lifespan=lifespan,
    )
    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        logger.info(
            "%s %s -> %d (%.0f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            extra={"request_id": request_id, "duration_ms": round(elapsed, 1)},
        )
        return response

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError) -> JSONResponse:
        logger.error("LLM unavailable: %s", exc)
        return JSONResponse(
            {"detail": "LLM provider unavailable, try again later"}, status_code=503
        )

    @app.exception_handler(VectorStoreError)
    async def store_error(_: Request, exc: VectorStoreError) -> JSONResponse:
        logger.error("Vector store error: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    app.include_router(router)
    return app
