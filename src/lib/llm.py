"""Minimal LLM client: tries each configured provider in turn."""

import os
import time
from functools import cache

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError
from openai.types.chat import ChatCompletionContentPartParam

from config import LLM_PROVIDERS, LLMProvider

# Per-minute limits clear quickly, so wait and retry before giving up on a provider.
RATE_LIMIT_RETRIES = 3
MAX_RETRY_WAIT_SECONDS = 15.0


@cache
def _client(provider: LLMProvider, api_key: str) -> OpenAI:
    return OpenAI(base_url=provider.base_url, api_key=api_key)


# A prompt is plain text, or a list of text and image parts for multimodal models.
Prompt = str | list[ChatCompletionContentPartParam]


def _ask(provider: LLMProvider, api_key: str, prompt: Prompt, max_tokens: int) -> str:
    response = _client(provider, api_key).chat.completions.create(
        model=provider.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    reply = (response.choices[0].message.content or "").strip()
    # Models often wrap a one-line answer in quotes even when told not to.
    if len(reply) > 1 and reply[0] == reply[-1] and reply[0] in "\"'":
        reply = reply[1:-1].strip()
    return reply


def _retry_wait(error: RateLimitError, attempt: int) -> float:
    """Seconds to wait before retrying: the server's Retry-After, else back off."""
    try:
        wait = float(error.response.headers["retry-after"])
    except (KeyError, ValueError):
        wait = 2.0 ** (attempt + 1)
    return min(wait + 0.5, MAX_RETRY_WAIT_SECONDS)


def _ask_with_retries(
    provider: LLMProvider, api_key: str, prompt: Prompt, max_tokens: int
) -> str:
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return _ask(provider, api_key, prompt, max_tokens)
        except RateLimitError as e:
            if attempt == RATE_LIMIT_RETRIES - 1:
                raise
            time.sleep(_retry_wait(e, attempt))
    raise AssertionError("unreachable")


def complete(prompt: Prompt, max_tokens: int = 100) -> str:
    """Send ``prompt`` to the first provider that answers and return the stripped reply.

    Sampling is deterministic and the reply is capped at ``max_tokens``, since every
    prompt asks for a short answer.
    """
    load_dotenv()
    failures: list[str] = []
    for provider in LLM_PROVIDERS:
        api_key = os.environ.get(provider.api_key_env)
        if not api_key:
            failures.append(f"{provider.name}: {provider.api_key_env} not set")
            continue
        try:
            return _ask_with_retries(provider, api_key, prompt, max_tokens)
        except OpenAIError as e:
            failures.append(f"{provider.name}: {e}")
    raise RuntimeError("LLM request failed. " + " | ".join(failures))
