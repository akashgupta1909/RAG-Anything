from typing import Any, cast

from rag.config import AgentConfig, GenerationConfig, Settings
from rag.llm import LLMError
from rag.models import RetrievedChunk
from rag.pipeline import ContextGrader, QueryPlanner, RAGPipeline, interleave
from tests.conftest import FakeLLM, make_chunk, plan_json


class FakeRetriever:
    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self.results = results
        self.calls: list[tuple[str, Any]] = []

    def search(
        self, query: str, *, domains: Any = None, **_: Any
    ) -> list[RetrievedChunk]:
        self.calls.append((query, domains))
        return self.results.get(query, [])


class FakeDiagrams:
    def for_chunks(self, chunks: Any, limit: int) -> list[Any]:
        return []


def _pipeline(
    settings: Settings, llm: FakeLLM, retriever: FakeRetriever, **agent: Any
) -> RAGPipeline:
    agent_config = AgentConfig(**agent)
    return RAGPipeline(
        retriever=cast(Any, retriever),
        diagrams=cast(Any, FakeDiagrams()),
        llm=cast(Any, llm),
        planner=QueryPlanner(cast(Any, llm), settings.domains, agent_config),
        grader=ContextGrader(cast(Any, llm)),
        agent=agent_config,
        generation=GenerationConfig(max_context_chunks=4),
    )


def test_interleave_round_robin_dedupes_and_limits() -> None:
    a = [make_chunk("a1"), make_chunk("shared"), make_chunk("a3")]
    b = [make_chunk("b1"), make_chunk("shared")]
    merged = interleave([a, b], limit=4)
    assert [c.chunk_id for c in merged] == ["a1", "b1", "shared", "a3"]


def test_planner_filters_unknown_domains_and_caps_sub_queries(
    settings: Settings,
) -> None:
    llm = FakeLLM([plan_json(["python", "cooking"], ["q1", "q2", "q3", "q4"])])
    plan = QueryPlanner(
        cast(Any, llm), settings.domains, AgentConfig(max_sub_queries=2)
    ).plan("question")
    assert plan.domains == ["python"]
    assert plan.sub_queries == ["q1", "q2"]
    assert plan.source == "llm"


def test_planner_falls_back_when_llm_fails(settings: Settings) -> None:
    llm = FakeLLM([LLMError("down")])
    plan = QueryPlanner(cast(Any, llm), settings.domains, AgentConfig()).plan(
        "what is a pod"
    )
    assert plan.source == "fallback"
    assert plan.domains == ["python", "kubernetes"]
    assert plan.sub_queries == ["what is a pod"]


def test_planner_respects_allowed_domains(settings: Settings) -> None:
    llm = FakeLLM([plan_json(["python", "kubernetes"], ["q"])])
    plan = QueryPlanner(cast(Any, llm), settings.domains, AgentConfig()).plan(
        "q", ["kubernetes"]
    )
    assert plan.domains == ["kubernetes"]


def test_multi_part_question_retrieves_each_sub_query(settings: Settings) -> None:
    question = "asyncio and pods"
    retriever = FakeRetriever(
        {
            question: [make_chunk("q0")],
            "asyncio tasks": [make_chunk("py1"), make_chunk("py2")],
            "kubernetes pods": [make_chunk("k1", domain="kubernetes")],
        }
    )
    llm = FakeLLM(
        [
            plan_json(["python", "kubernetes"], ["asyncio tasks", "kubernetes pods"]),
            "Answer [1][3].",
        ]
    )
    response = _pipeline(settings, llm, retriever, self_correction=False).answer(
        question
    )

    assert [q for q, _ in retriever.calls] == [
        question,
        "asyncio tasks",
        "kubernetes pods",
    ]
    assert [c.title for c in response.citations] == [
        "Title q0",
        "Title py1",
        "Title k1",
        "Title py2",
    ]
    assert [c.id for c in response.citations if c.cited] == [1, 3]
    assert [s.name for s in response.trace.steps] == ["plan", "retrieve", "generate"]


def test_corrective_round_runs_when_context_is_insufficient(settings: Settings) -> None:
    retriever = FakeRetriever(
        {
            "q": [make_chunk("weak")],
            "better query": [make_chunk("good")],
        }
    )
    llm = FakeLLM(
        [
            plan_json(["python"], ["q"]),
            '{"sufficient": false, "missing": "details", "follow_up_query": "better query"}',
            '{"sufficient": true}',
            "Final [2].",
        ]
    )
    response = _pipeline(settings, llm, retriever, max_retrieval_rounds=3).answer("q")
    assert response.rounds == 2
    # The corrective search is not domain-filtered (the router may have been wrong).
    assert retriever.calls[-1] == ("better query", None)
    assert [c.chunk_id for c in response.context] == ["weak", "good"]
    assert [s.name for s in response.trace.steps] == [
        "plan",
        "retrieve",
        "grade",
        "corrective_retrieve",
        "grade",
        "generate",
    ]


def test_grader_failure_fails_open(settings: Settings) -> None:
    retriever = FakeRetriever({"q": [make_chunk("c")]})
    llm = FakeLLM([plan_json(["python"], ["q"]), LLMError("grader down"), "ok"])
    response = _pipeline(settings, llm, retriever).answer("q")
    assert response.rounds == 1
    assert response.answer == "ok"


def test_stream_emits_events_in_order(settings: Settings) -> None:
    retriever = FakeRetriever({"q": [make_chunk("c")]})
    llm = FakeLLM(
        [plan_json(["python"], ["q"]), '{"sufficient": true}', "streamed answer [1]"]
    )
    events = list(_pipeline(settings, llm, retriever).stream("q"))
    types = [e["type"] for e in events]
    assert types[:3] == ["plan", "sources", "diagrams"]
    assert types[-1] == "done" and events[-1]["cited"] == [1]
    assert (
        "".join(e["text"] for e in events if e["type"] == "token")
        == "streamedanswer[1]"
    )


def test_cited_ids_accepts_bracket_and_prose_styles() -> None:
    from rag.pipeline.rag import cited_ids

    assert cited_ids("A [1] and B [2, 4][5].") == {1, 2, 4, 5}
    assert cited_ids("Per source 3 and sources 5-7.") == {3, 5, 6, 7}
    assert cited_ids("See sources 1, 8 and 9.") == {1, 8, 9}
    assert cited_ids("No citations, 2024 [draft].") == set()
