import pytest
from pydantic import BaseModel

from rag.config import LLMConfig, ProviderConfig
from rag.llm import LLMClient, LLMError, parse_json_model, strip_reasoning
from rag.llm.client import _ThinkFilter


class Plan(BaseModel):
    domains: list[str]


def test_parse_json_tolerates_fences_prose_and_reasoning() -> None:
    reply = '<think>let me think {"no": 1}</think>Sure!\n```json\n{"domains": ["python"]}\n```'
    assert parse_json_model(reply, Plan).domains == ["python"]


def test_parse_json_rejects_invalid() -> None:
    with pytest.raises(LLMError):
        parse_json_model("no json here", Plan)
    with pytest.raises(LLMError):
        parse_json_model('{"domains": "not-a-list"}', Plan)


def test_strip_reasoning_handles_unclosed_block() -> None:
    assert strip_reasoning("<think>never closed") == ""
    assert strip_reasoning("<think>x</think> answer ") == "answer"


def test_think_filter_on_split_stream() -> None:
    f = _ThinkFilter()
    tokens = ["Hel", "lo <thi", "nk>secret", " stuff</th", "ink> world", " <", "b>"]
    out = "".join(f.feed(t) for t in tokens) + f.flush()
    assert out == "Hello  world <b>"


def test_missing_keys_give_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAKE_KEY", raising=False)
    client = LLMClient(
        LLMConfig(),
        [
            ProviderConfig(
                name="p",
                base_url="http://localhost:1",
                api_key_env="FAKE_KEY",
                model="m",
            )
        ],
    )
    assert client.available == []
    with pytest.raises(LLMError, match="FAKE_KEY not set"):
        client.complete([{"role": "user", "content": "hi"}])
