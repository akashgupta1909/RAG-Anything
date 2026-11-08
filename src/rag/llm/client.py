"""LLM client over OpenAI-compatible APIs, with ordered provider fallback.

Providers are tried in configured order: one without an API key is skipped,
and one that errors (rate limit after retries, outage, ...) falls through to
the next. Structured outputs are parsed into Pydantic models.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from functools import cached_property
from typing import TypeVar

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from rag.config import LLMConfig, ProviderConfig
from rag.utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)
Messages = list[ChatCompletionMessageParam]

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMError(RuntimeError):
    """No provider could produce a (valid) response."""


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks some reasoning models emit inline."""
    return _THINK_RE.sub("", text).strip()


def parse_json_model[T: BaseModel](text: str, model: type[T]) -> T:
    """Parse ``text`` into ``model``, tolerating code fences and surrounding prose."""
    cleaned = strip_reasoning(text)
    match = _JSON_OBJECT_RE.search(cleaned)
    candidate = match.group() if match else cleaned
    try:
        return model.model_validate(json.loads(candidate))
    except (json.JSONDecodeError, ValidationError) as e:
        raise LLMError(
            f"invalid {model.__name__} JSON from LLM: {e}; reply={text[:300]!r}"
        ) from e


class _Provider:
    def __init__(self, config: ProviderConfig, llm: LLMConfig) -> None:
        self.config = config
        self.llm = llm

    @cached_property
    def client(self) -> OpenAI:
        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.llm.timeout_seconds,
            max_retries=self.llm.max_retries,  # SDK backs off on 429/5xx
        )


class LLMClient:
    def __init__(
        self, config: LLMConfig, providers: list[ProviderConfig] | None = None
    ) -> None:
        self.config = config
        self._providers = [
            _Provider(p, config) for p in (providers or config.providers)
        ]
        if not self._providers:
            raise LLMError("no LLM providers configured")

    @property
    def available(self) -> list[str]:
        """Names of providers that have an API key."""
        return [p.config.name for p in self._providers if p.config.api_key]

    def complete(
        self,
        messages: Messages,
        *,
        max_tokens: int = 1000,
        temperature: float | None = None,
    ) -> str:
        """Return the first successful completion, with reasoning blocks removed."""
        failures: list[str] = []
        for provider in self._providers:
            if not provider.config.api_key:
                failures.append(
                    f"{provider.config.name}: {provider.config.api_key_env} not set"
                )
                continue
            try:
                response = provider.client.chat.completions.create(
                    model=provider.config.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.config.temperature
                    if temperature is None
                    else temperature,
                    extra_body=provider.config.extra_body or None,
                )
            except OpenAIError as e:
                logger.warning("LLM provider %s failed: %s", provider.config.name, e)
                failures.append(f"{provider.config.name}: {e}")
                continue
            text = strip_reasoning(response.choices[0].message.content or "")
            if text:
                usage = response.usage
                logger.debug(
                    "LLM %s ok",
                    provider.config.name,
                    extra={
                        "provider": provider.config.name,
                        "prompt_tokens": usage.prompt_tokens if usage else None,
                        "completion_tokens": usage.completion_tokens if usage else None,
                    },
                )
                return text
            failures.append(f"{provider.config.name}: empty reply")
        raise LLMError("all LLM providers failed: " + " | ".join(failures))

    def complete_json(
        self, messages: Messages, model: type[T], *, max_tokens: int = 800
    ) -> T:
        """Complete and parse the reply into ``model`` (one corrective retry)."""
        reply = self.complete(messages, max_tokens=max_tokens)
        try:
            return parse_json_model(reply, model)
        except LLMError as first_error:
            retry: Messages = [
                *messages,
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That was not valid JSON for the schema ({first_error}). "
                    "Reply with only the corrected JSON object.",
                },
            ]
            return parse_json_model(self.complete(retry, max_tokens=max_tokens), model)

    def stream(self, messages: Messages, *, max_tokens: int = 1500) -> Iterator[str]:
        """Yield text deltas. Falls back to the next provider only before the first token."""
        failures: list[str] = []
        for provider in self._providers:
            if not provider.config.api_key:
                continue
            started = False
            try:
                stream = provider.client.chat.completions.create(
                    model=provider.config.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.config.temperature,
                    extra_body=provider.config.extra_body or None,
                    stream=True,
                )
                filtered = _ThinkFilter()
                for event in stream:
                    delta = event.choices[0].delta.content if event.choices else None
                    if delta and (text := filtered.feed(delta)):
                        started = True
                        yield text
                if tail := filtered.flush():
                    yield tail
                return
            except OpenAIError as e:
                if (
                    started
                ):  # the client already has partial output; don't mix providers
                    raise LLMError(
                        f"{provider.config.name} failed mid-stream: {e}"
                    ) from e
                logger.warning(
                    "LLM provider %s failed while streaming: %s",
                    provider.config.name,
                    e,
                )
                failures.append(f"{provider.config.name}: {e}")
        raise LLMError("all LLM providers failed: " + " | ".join(failures))


class _ThinkFilter:
    """Drops ``<think>...</think>`` content from a token stream."""

    def __init__(self) -> None:
        self.buffer = ""
        self.thinking = False

    def feed(self, delta: str) -> str:
        self.buffer += delta
        out = ""
        while self.buffer:
            if self.thinking:
                end = self.buffer.find("</think>")
                if end == -1:
                    self.buffer = self.buffer[-8:]
                    return out
                self.buffer = self.buffer[end + len("</think>") :]
                self.thinking = False
            else:
                start = self.buffer.find("<think>")
                if start == -1:
                    # Hold back a possible partial "<think>" tag at the end.
                    lt = self.buffer.rfind("<", max(0, len(self.buffer) - 7))
                    partial = lt != -1 and "<think>".startswith(self.buffer[lt:])
                    safe = lt if partial else len(self.buffer)
                    out += self.buffer[:safe]
                    self.buffer = self.buffer[safe:]
                    return out
                out += self.buffer[:start]
                self.buffer = self.buffer[start + len("<think>") :]
                self.thinking = True
        return out

    def flush(self) -> str:
        """Return any held-back text once the stream has ended."""
        tail = "" if self.thinking else self.buffer
        self.buffer = ""
        return tail
