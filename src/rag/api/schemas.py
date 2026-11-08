"""Request and response models of the REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rag.models import Diagram, RetrievedChunk


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
        examples=["How does a Kubernetes Service load-balance traffic to Pods?"],
    )
    domains: list[str] | None = Field(
        default=None, description="Restrict retrieval to these domains."
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    domains: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["hybrid", "dense", "sparse"] | None = None
    rerank: bool | None = None


class SearchResponse(BaseModel):
    results: list[RetrievedChunk]


class DiagramSearchRequest(BaseModel):
    query: str = Field(
        min_length=1, max_length=500, examples=["cache in front of a database"]
    )
    domains: list[str] | None = None
    limit: int = Field(default=5, ge=1, le=20)


class DiagramSearchResponse(BaseModel):
    results: list[Diagram]


class DomainInfo(BaseModel):
    name: str
    description: str
    documents: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    vectordb: bool
    llm_providers: list[str]
    version: str
