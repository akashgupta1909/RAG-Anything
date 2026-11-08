"""The LLM-driven decision steps of the pipeline: planning and context grading.

Both steps fail open: if the LLM is unavailable or returns something invalid,
the pipeline falls back to plain retrieval rather than failing the request.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from rag.config import AgentConfig, DomainConfig
from rag.llm import LLMClient, LLMError
from rag.models import RetrievedChunk
from rag.prompts import grader_messages, planner_messages
from rag.utils import get_logger

logger = get_logger(__name__)


class QueryPlan(BaseModel):
    domains: list[str]
    sub_queries: list[str] = Field(default_factory=list)
    needs_diagrams: bool = False
    source: Literal["llm", "fallback"] = "llm"

    @field_validator("sub_queries")
    @classmethod
    def _clean_queries(cls, queries: list[str]) -> list[str]:
        return [q.strip() for q in queries if q and q.strip()]


class ContextGrade(BaseModel):
    sufficient: bool
    missing: str = ""
    follow_up_query: str = ""


class QueryPlanner:
    """Routes a question to knowledge domains and splits multi-part questions."""

    def __init__(
        self, llm: LLMClient, domains: Sequence[DomainConfig], config: AgentConfig
    ) -> None:
        self.llm = llm
        self.domains = list(domains)
        self.config = config

    def plan(
        self, question: str, allowed_domains: Sequence[str] | None = None
    ) -> QueryPlan:
        candidates = [
            d for d in self.domains if not allowed_domains or d.name in allowed_domains
        ]
        fallback = QueryPlan(
            domains=[d.name for d in candidates],
            sub_queries=[question],
            source="fallback",
        )
        if not self.config.planning or not candidates:
            return fallback
        if len(candidates) == 1 and self.config.max_sub_queries == 1:
            return fallback  # nothing to decide

        try:
            plan = self.llm.complete_json(
                planner_messages(question, candidates, self.config.max_sub_queries),
                QueryPlan,
                max_tokens=400,
            )
        except LLMError as e:
            logger.warning("Planner failed, using fallback plan: %s", e)
            return fallback

        known = {d.name for d in candidates}
        domains = [d for d in dict.fromkeys(plan.domains) if d in known]
        sub_queries = plan.sub_queries[: self.config.max_sub_queries] or [question]
        return QueryPlan(
            # An empty or unknown routing decision must not hide all documents.
            domains=domains or sorted(known),
            sub_queries=sub_queries,
            needs_diagrams=plan.needs_diagrams,
            source="llm",
        )


class ContextGrader:
    """Judges whether retrieved context can answer the question (CRAG-style)."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def grade(self, question: str, chunks: Sequence[RetrievedChunk]) -> ContextGrade:
        if not chunks:
            return ContextGrade(
                sufficient=False, missing="no sources", follow_up_query=question
            )
        try:
            return self.llm.complete_json(
                grader_messages(question, chunks), ContextGrade, max_tokens=300
            )
        except LLMError as e:
            logger.warning("Grader failed, assuming context is sufficient: %s", e)
            return ContextGrade(sufficient=True)
