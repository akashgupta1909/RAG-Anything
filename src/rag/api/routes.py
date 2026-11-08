"""HTTP endpoints. Handlers are sync functions: FastAPI runs them in a thread
pool, which suits the CPU-bound embedding/reranking and blocking LLM calls."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from rag import __version__
from rag.api.schemas import (
    AskRequest,
    DiagramSearchRequest,
    DiagramSearchResponse,
    DomainInfo,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from rag.container import Container
from rag.llm import LLMError
from rag.pipeline import RAGResponse
from rag.utils import get_logger

logger = get_logger(__name__)

MAX_IMAGE_BYTES = 5 * 1024 * 1024

router = APIRouter()


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


def _check_domains(container: Container, domains: list[str] | None) -> None:
    unknown = set(domains or []) - set(container.settings.domain_names)
    if unknown:
        raise HTTPException(
            422,
            f"unknown domains {sorted(unknown)}; available: {container.settings.domain_names}",
        )


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(container: ContainerDep) -> HealthResponse:
    vectordb = container.store.healthy()
    providers = container.llm.available
    return HealthResponse(
        status="ok" if vectordb and providers else "degraded",
        vectordb=vectordb,
        llm_providers=providers,
        version=__version__,
    )


@router.get("/v1/domains", response_model=list[DomainInfo], tags=["rag"])
def domains(container: ContainerDep) -> list[DomainInfo]:
    counts = container.store.stats().get("documents", {})
    return [
        DomainInfo(
            name=d.name, description=d.description, documents=counts.get(d.name, 0)
        )
        for d in container.settings.domains
    ]


@router.post("/v1/ask", response_model=RAGResponse, tags=["rag"])
def ask(body: AskRequest, container: ContainerDep) -> RAGResponse:
    """Answer a question: plan, retrieve, self-check, generate with citations."""
    _check_domains(container, body.domains)
    return container.pipeline.answer(body.question, domains=body.domains)


@router.post("/v1/ask/stream", tags=["rag"])
def ask_stream(body: AskRequest, container: ContainerDep) -> StreamingResponse:
    """Server-Sent Events: ``plan``, ``sources``, ``diagrams``, ``token``..., ``done``."""
    _check_domains(container, body.domains)

    def events() -> Iterator[str]:
        try:
            for event in container.pipeline.stream(body.question, domains=body.domains):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        except LLMError as e:
            logger.error("Streaming failed: %s", e)
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/v1/search", response_model=SearchResponse, tags=["retrieval"])
def search(body: SearchRequest, container: ContainerDep) -> SearchResponse:
    """Retrieval only (no LLM): hybrid search + rerank."""
    _check_domains(container, body.domains)
    results = container.retriever.search(
        body.query,
        domains=body.domains,
        top_k=body.top_k,
        mode=body.mode,
        rerank=body.rerank,
    )
    return SearchResponse(results=results)


@router.post(
    "/v1/diagrams/search", response_model=DiagramSearchResponse, tags=["multimodal"]
)
def diagrams_by_text(
    body: DiagramSearchRequest, container: ContainerDep
) -> DiagramSearchResponse:
    """Find architecture diagrams matching a text description (CLIP)."""
    _check_domains(container, body.domains)
    return DiagramSearchResponse(
        results=container.diagram_search.by_text(
            body.query, domains=body.domains, limit=body.limit
        )
    )


@router.post(
    "/v1/diagrams/search-by-image",
    response_model=DiagramSearchResponse,
    tags=["multimodal"],
)
def diagrams_by_image(
    container: ContainerDep,
    image: Annotated[UploadFile, File(description="PNG/JPEG/GIF/WebP, max 5 MB")],
    limit: int = 5,
) -> DiagramSearchResponse:
    """Find indexed diagrams visually similar to an uploaded image (CLIP)."""
    data = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image larger than 5 MB")
    suffix = Path(image.filename or "upload").suffix.lower() or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            with Image.open(tmp.name) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(415, "not a valid image") from e
        results = container.diagram_search.by_image(
            tmp.name, limit=max(1, min(limit, 20))
        )
    return DiagramSearchResponse(results=results)
