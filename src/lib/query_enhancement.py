"""Improve search queries with an LLM before they are searched."""

import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from config import ENHANCEMENT_MODEL, OPENROUTER_BASE_URL

ENHANCEMENT_METHODS = ("spell", "rewrite", "expand")


def _spell_prompt(query: str) -> str:
    return f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""


def _rewrite_prompt(query: str) -> str:
    return f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""


def _expand_prompt(query: str) -> str:
    return f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""


def _complete(prompt: str) -> str:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=ENHANCEMENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except OpenAIError as e:
        raise RuntimeError(f"Query enhancement request failed: {e}") from e
    return (response.choices[0].message.content or "").strip()


def enhance_query(query: str, method: str) -> str:
    """Return ``query`` improved by ``method``, or unchanged if the LLM gives nothing."""
    match method:
        case "spell":
            return _complete(_spell_prompt(query)) or query
        case "rewrite":
            return _complete(_rewrite_prompt(query)) or query
        case "expand":
            # The LLM returns only the extra terms; keep the original query in front.
            extra_terms = _complete(_expand_prompt(query))
            return f"{query} {extra_terms}" if extra_terms else query
        case _:
            raise ValueError(f"unknown enhancement method: {method!r}")
