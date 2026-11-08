"""Evaluation harness: retrieval quality (ranked metrics) and answer quality (LLM judge).

Retrieval runs are cheap and deterministic, so they are the main regression
signal and support ablations (``mode`` = dense / sparse / hybrid, reranking on
or off, planner on or off). Generation runs call the LLM and a separate judge
model, and report faithfulness and answer relevance on a 1-5 scale.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.config import RetrievalMode
from rag.evaluation.metrics import percentile, retrieval_scores
from rag.llm import LLMClient, LLMError
from rag.pipeline import RAGPipeline
from rag.prompts import PROMPT_VERSION, faithfulness_messages, relevance_messages
from rag.retrieval import Retriever
from rag.utils import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int], None]


class GoldenExample(BaseModel):
    id: str
    question: str
    relevant_urls: list[str]
    domains: list[str] = Field(default_factory=list)
    category: str = "factual"
    reference_answer: str | None = None


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = ""


def load_golden_set(path: Path) -> list[GoldenExample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenExample.model_validate(item) for item in data["examples"]]


def _aggregate(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    return {key: round(mean(r[key] for r in rows), 4) for key in keys if rows}


def evaluate_retrieval(
    examples: list[GoldenExample],
    *,
    retriever: Retriever,
    pipeline: RAGPipeline | None = None,
    k: int = 5,
    mode: RetrievalMode | None = None,
    rerank: bool | None = None,
    strategy: Literal["single", "agentic"] = "single",
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Score retrieval on every example.

    ``single`` searches the question once (no LLM). ``agentic`` runs the full
    retrieval stage of the pipeline: planning, sub-queries, corrective search.
    """
    if strategy == "agentic" and pipeline is None:
        raise ValueError("agentic strategy needs the pipeline")
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for i, example in enumerate(examples, start=1):
        start = time.perf_counter()
        if strategy == "agentic" and pipeline is not None:
            chunks = pipeline.retrieve(example.question).chunks
        else:
            chunks = retriever.search(
                example.question, top_k=k, mode=mode, rerank=rerank
            )
        latencies.append((time.perf_counter() - start) * 1000)
        scores = retrieval_scores([c.url for c in chunks], example.relevant_urls, k)
        rows.append(
            {
                "id": example.id,
                "category": example.category,
                "question": example.question,
                **scores,
                "retrieved": list(dict.fromkeys(c.url for c in chunks))[:k],
            }
        )
        if on_progress:
            on_progress(i, len(examples))

    metric_keys = [f"hit@{k}", f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}"]
    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({r["category"] for r in rows}):
        by_category[category] = _aggregate(
            [r for r in rows if r["category"] == category], metric_keys
        )
    return {
        "kind": "retrieval",
        "config": {
            "k": k,
            "mode": mode or retriever.config.mode,
            "rerank": retriever.config.rerank.enabled if rerank is None else rerank,
            "strategy": strategy,
        },
        "summary": {
            **_aggregate(rows, metric_keys),
            "latency_ms_p50": round(percentile(latencies, 50), 1),
            "latency_ms_p95": round(percentile(latencies, 95), 1),
        },
        "by_category": by_category,
        "examples": rows,
    }


def evaluate_generation(
    examples: list[GoldenExample],
    *,
    pipeline: RAGPipeline,
    judge: LLMClient,
    k: int = 5,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the full pipeline and grade each answer with the judge model."""
    rows: list[dict[str, Any]] = []
    for i, example in enumerate(examples, start=1):
        try:
            response = pipeline.answer(example.question)
            faithfulness = judge.complete_json(
                faithfulness_messages(response.answer, response.context),
                JudgeScore,
                max_tokens=600,
            )
            relevance = judge.complete_json(
                relevance_messages(
                    example.question, response.answer, example.reference_answer
                ),
                JudgeScore,
                max_tokens=600,
            )
        except LLMError as e:
            logger.warning("Generation eval failed for %s: %s", example.id, e)
            rows.append({"id": example.id, "error": str(e)})
            continue
        cited = [c for c in response.citations if c.cited]
        rows.append(
            {
                "id": example.id,
                "category": example.category,
                "question": example.question,
                "faithfulness": faithfulness.score,
                "faithfulness_reason": faithfulness.reason,
                "relevance": relevance.score,
                "relevance_reason": relevance.reason,
                "citations": len(cited),
                "has_citation": 1.0 if cited else 0.0,
                "rounds": response.rounds,
                "latency_ms": response.trace.total_ms,
                **retrieval_scores(
                    [c.url for c in response.citations], example.relevant_urls, k
                ),
                "answer": response.answer,
            }
        )
        if on_progress:
            on_progress(i, len(examples))

    ok = [r for r in rows if "error" not in r]
    latencies = [r["latency_ms"] for r in ok]
    return {
        "kind": "generation",
        "config": {"k": k, "prompt_version": PROMPT_VERSION},
        "summary": {
            **_aggregate(
                ok, ["faithfulness", "relevance", "has_citation", "rounds", f"hit@{k}"]
            ),
            "errors": len(rows) - len(ok),
            "latency_ms_p50": round(percentile(latencies, 50), 1),
            "latency_ms_p95": round(percentile(latencies, 95), 1),
        },
        "examples": rows,
    }


def save_report(report: dict[str, Any], reports_dir: Path, label: str) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = reports_dir / f"{stamp}_{label}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
