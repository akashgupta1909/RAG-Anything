from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.config import Settings
from rag.container import Container
from rag.pipeline import QueryPlan, RAGResponse
from rag.utils import Trace
from tests.conftest import make_chunk


class FakePipeline:
    def answer(self, question: str, *, domains: Any = None) -> RAGResponse:
        return RAGResponse(
            question=question,
            answer="A Pod is ... [1]",
            citations=[],
            diagrams=[],
            plan=QueryPlan(domains=domains or ["kubernetes"], sub_queries=[question]),
            rounds=1,
            trace=Trace(),
        )

    def stream(self, question: str, *, domains: Any = None) -> Iterator[dict[str, Any]]:
        yield {"type": "token", "text": "hi"}
        yield {"type": "done", "cited": [], "trace": {}}


class FakeRetriever:
    def search(self, query: str, **_: Any) -> list[Any]:
        return [make_chunk("c1", text=query)]


class FakeStore:
    def healthy(self) -> bool:
        return True

    def stats(self) -> dict[str, Any]:
        return {"documents": {"python": 3}}

    def close(self) -> None:
        pass


class FakeLLM:
    def __init__(self) -> None:
        self.available = ["fake"]


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    container = Container(settings)
    # cached_property values can be pre-seeded with fakes.
    container.__dict__.update(
        pipeline=FakePipeline(),
        retriever=FakeRetriever(),
        store=FakeStore(),
        llm=FakeLLM(),
    )
    with TestClient(create_app(container=container, warm_up=False)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "x-request-id" in response.headers


def test_ask(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"question": "What is a pod?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("A Pod")
    assert "context" not in body  # internal field is not serialised


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/v1/ask", json={"question": ""}).status_code == 422
    unknown = client.post("/v1/ask", json={"question": "hello", "domains": ["cooking"]})
    assert unknown.status_code == 422


def test_ask_stream_is_sse(client: TestClient) -> None:
    response = client.post("/v1/ask/stream", json={"question": "What is a pod?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in response.text and "event: done" in response.text


def test_search_and_domains(client: TestClient) -> None:
    hits = client.post("/v1/search", json={"query": "pods", "top_k": 3}).json()[
        "results"
    ]
    assert hits[0]["chunk_id"] == "c1"
    domains = client.get("/v1/domains").json()
    assert {d["name"]: d["documents"] for d in domains} == {
        "python": 3,
        "kubernetes": 0,
    }
