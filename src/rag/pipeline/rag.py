"""The RAG pipeline: plan -> retrieve -> grade -> (corrective search) -> answer.

The control flow is a bounded loop with explicit steps rather than an
open-ended agent: every request does at most ``max_retrieval_rounds``
retrievals, so latency and cost stay predictable, and each step is recorded
in a ``Trace`` returned to the caller.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any

from pydantic import BaseModel, Field

from rag.config import AgentConfig, GenerationConfig
from rag.llm import LLMClient
from rag.models import Diagram, RetrievedChunk
from rag.pipeline.steps import ContextGrade, ContextGrader, QueryPlan, QueryPlanner
from rag.prompts import PROMPT_VERSION, answer_messages
from rag.retrieval import DiagramSearch, Retriever
from rag.utils import Trace, get_logger

logger = get_logger(__name__)

# "[3]", "[2, 5]", and the "source 4" / "sources 1-2" style some fallback models use.
_CITATION_RE = re.compile(
    r"\[(\d+(?:\s*[,;]\s*\d+)*)\]|\bsources?\s+(\d+(?:\s*(?:[,\u2013-]|and)\s*\d+)*)",
    re.IGNORECASE,
)


def cited_ids(answer: str) -> set[int]:
    """Source ids an answer cites; ranges such as "1-3" are expanded."""
    ids: set[int] = set()
    for bracket, prose in _CITATION_RE.findall(answer):
        text = bracket or prose
        for part in re.split(r"\s*(?:[,;]|and)\s*", text):
            bounds = [
                int(n)
                for n in re.split(r"\s*[\u2013-]\s*", part)
                if n.strip().isdigit()
            ]
            if len(bounds) == 2 and bounds[0] <= bounds[1] <= bounds[0] + 20:
                ids.update(range(bounds[0], bounds[1] + 1))
            else:
                ids.update(bounds)
    return ids


class Citation(BaseModel):
    id: int
    title: str
    heading: str
    url: str
    domain: str
    snippet: str
    rerank_score: float | None = None
    cited: bool = False


class RetrievalResult(BaseModel):
    plan: QueryPlan
    chunks: list[RetrievedChunk]
    grades: list[ContextGrade] = Field(default_factory=list)
    rounds: int = 1


class RAGResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    diagrams: list[Diagram]
    plan: QueryPlan
    rounds: int
    trace: Trace
    prompt_version: str = PROMPT_VERSION
    # Full context passed to the generator (for evaluation); not serialised.
    context: list[RetrievedChunk] = Field(default_factory=list, exclude=True)


def interleave(
    result_lists: Sequence[Sequence[RetrievedChunk]], limit: int
) -> list[RetrievedChunk]:
    """Round-robin merge of ranked lists, dropping duplicates.

    Taking the best chunk of each sub-query first guarantees every part of a
    multi-part question is represented in the context, which a single global
    sort by score would not.
    """
    merged: dict[str, RetrievedChunk] = {}
    depth = max((len(r) for r in result_lists), default=0)
    for rank in range(depth):
        for results in result_lists:
            if rank < len(results) and results[rank].chunk_id not in merged:
                merged[results[rank].chunk_id] = results[rank]
                if len(merged) == limit:
                    return list(merged.values())
    return list(merged.values())


def build_citations(
    chunks: Sequence[RetrievedChunk], answer: str = ""
) -> list[Citation]:
    cited = cited_ids(answer)
    return [
        Citation(
            id=i,
            title=c.title,
            heading=" > ".join(c.heading_path),
            url=c.url,
            domain=c.domain,
            snippet=c.text[:300],
            rerank_score=c.rerank_score,
            cited=i in cited,
        )
        for i, c in enumerate(chunks, start=1)
    ]


class RAGPipeline:
    def __init__(
        self,
        *,
        retriever: Retriever,
        diagrams: DiagramSearch,
        llm: LLMClient,
        planner: QueryPlanner,
        grader: ContextGrader,
        agent: AgentConfig,
        generation: GenerationConfig,
    ) -> None:
        self.retriever = retriever
        self.diagrams = diagrams
        self.llm = llm
        self.planner = planner
        self.grader = grader
        self.agent = agent
        self.generation = generation

    # -- retrieval ---------------------------------------------------------
    def retrieve(
        self,
        question: str,
        *,
        domains: Sequence[str] | None = None,
        trace: Trace | None = None,
    ) -> RetrievalResult:
        """Plan, retrieve for each sub-query, and run corrective rounds if needed."""
        trace = trace or Trace()
        limit = self.generation.max_context_chunks

        with trace.step("plan") as info:
            plan = self.planner.plan(question, domains)
            info.update(plan.model_dump())

        # The original question is always searched too: it guards against a
        # sub-query rewrite that drifts from what the user asked.
        queries = list(dict.fromkeys([question, *plan.sub_queries]))
        with trace.step("retrieve", queries=queries, domains=plan.domains) as info:
            per_query = [
                self.retriever.search(q, domains=plan.domains) for q in queries
            ]
            chunks = interleave(per_query, limit)
            info["chunks"] = len(chunks)

        result = RetrievalResult(plan=plan, chunks=chunks)
        if not self.agent.self_correction:
            return result

        while result.rounds < self.agent.max_retrieval_rounds:
            with trace.step("grade", round=result.rounds) as info:
                grade = self.grader.grade(question, result.chunks)
                result.grades.append(grade)
                info.update(grade.model_dump())
            if grade.sufficient:
                break
            follow_up = grade.follow_up_query or question
            # Widen the search to every domain: the router may have been wrong.
            with trace.step("corrective_retrieve", query=follow_up) as info:
                extra = self.retriever.search(follow_up)
                result.chunks = interleave([result.chunks, extra], limit)
                result.rounds += 1
                info["chunks"] = len(result.chunks)
        return result

    # -- generation --------------------------------------------------------
    def _diagrams_for(self, chunks: Sequence[RetrievedChunk]) -> list[Diagram]:
        if self.generation.max_diagrams == 0:
            return []
        return self.diagrams.for_chunks(chunks, self.generation.max_diagrams)

    def answer(
        self, question: str, *, domains: Sequence[str] | None = None
    ) -> RAGResponse:
        trace = Trace()
        retrieval = self.retrieve(question, domains=domains, trace=trace)
        diagrams = self._diagrams_for(retrieval.chunks)
        with trace.step("generate") as info:
            answer = self.llm.complete(
                answer_messages(
                    question, retrieval.chunks, [d.caption for d in diagrams]
                ),
                max_tokens=self.generation.max_tokens,
            )
            info["answer_chars"] = len(answer)
        logger.info(
            "Answered question",
            extra={
                "question": question,
                "rounds": retrieval.rounds,
                "total_ms": trace.total_ms,
            },
        )
        return RAGResponse(
            question=question,
            answer=answer,
            citations=build_citations(retrieval.chunks, answer),
            diagrams=diagrams,
            plan=retrieval.plan,
            rounds=retrieval.rounds,
            trace=trace,
            context=retrieval.chunks,
        )

    def stream(
        self, question: str, *, domains: Sequence[str] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield events: ``plan``, ``sources``, ``diagrams``, many ``token``, then ``done``."""
        trace = Trace()
        retrieval = self.retrieve(question, domains=domains, trace=trace)
        diagrams = self._diagrams_for(retrieval.chunks)
        yield {
            "type": "plan",
            "plan": retrieval.plan.model_dump(),
            "rounds": retrieval.rounds,
        }
        yield {
            "type": "sources",
            "citations": [c.model_dump() for c in build_citations(retrieval.chunks)],
        }
        yield {"type": "diagrams", "diagrams": [d.model_dump() for d in diagrams]}
        parts: list[str] = []
        with trace.step("generate") as info:
            for token in self.llm.stream(
                answer_messages(
                    question, retrieval.chunks, [d.caption for d in diagrams]
                ),
                max_tokens=self.generation.max_tokens,
            ):
                parts.append(token)
                yield {"type": "token", "text": token}
            info["answer_chars"] = sum(len(p) for p in parts)
        cited = [
            c.id for c in build_citations(retrieval.chunks, "".join(parts)) if c.cited
        ]
        yield {"type": "done", "cited": cited, "trace": trace.model_dump()}
