"""Command-line interface for retrieval augmented generation (RAG)."""

import argparse
import sys

from config import RRF_K
from lib.hybrid_search import RRFResult, rrf_search
from lib.llm import complete

SEARCH_RESULT_COUNT = 5
# Answers are a few paragraphs, with headroom for models that "think" first.
ANSWER_MAX_TOKENS = 1500


def _format_documents(results: list[RRFResult]) -> str:
    return "\n\n".join(
        f"{result.movie['title']}: {result.movie['description']}" for result in results
    )


def _rag_prompt(query: str, results: list[RRFResult]) -> str:
    docs = _format_documents(results)
    return f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""


def _summarize_prompt(query: str, results: list[RRFResult]) -> str:
    return f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{_format_documents(results)}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""  # noqa: RUF001


def _citations_prompt(query: str, results: list[RRFResult]) -> str:
    # Number the documents so the answer can cite them as [1], [2], ...
    documents = "\n\n".join(
        f"[{position}] {result.movie['title']}: {result.movie['description']}"
        for position, result in enumerate(results, start=1)
    )
    return f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""


def _question_prompt(question: str, results: list[RRFResult]) -> str:
    return f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{_format_documents(results)}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")
    return number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize the search results for a query"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=SEARCH_RESULT_COUNT,
        help="Number of search results to summarize",
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Answer a query with citations to the search results"
    )
    citations_parser.add_argument("query", type=str, help="Search query to answer")
    citations_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=SEARCH_RESULT_COUNT,
        help="Number of search results to cite from",
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a question using the search results"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=SEARCH_RESULT_COUNT,
        help="Number of search results to answer from",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        match args.command:
            case "rag":
                query: str = args.query
                results = rrf_search(query, RRF_K, SEARCH_RESULT_COUNT)
                answer = complete(
                    _rag_prompt(query, results), max_tokens=ANSWER_MAX_TOKENS
                )
                print("Search Results:")
                for result in results:
                    print(f"- {result.movie['title']}")
                print("\nRAG Response:")
                print(answer)
            case "summarize":
                query = args.query
                results = rrf_search(query, RRF_K, args.limit)
                summary = complete(
                    _summarize_prompt(query, results), max_tokens=ANSWER_MAX_TOKENS
                )
                print("Search Results:")
                for result in results:
                    print(f"  - {result.movie['title']}")
                print("\nLLM Summary:")
                print(summary)
            case "citations":
                query = args.query
                results = rrf_search(query, RRF_K, args.limit)
                answer = complete(
                    _citations_prompt(query, results), max_tokens=ANSWER_MAX_TOKENS
                )
                print("Search Results:")
                for result in results:
                    print(f"  - {result.movie['title']}")
                print("\nLLM Answer:")
                print(answer)
            case "question":
                question: str = args.question
                results = rrf_search(question, RRF_K, args.limit)
                answer = complete(
                    _question_prompt(question, results), max_tokens=ANSWER_MAX_TOKENS
                )
                print("Search Results:")
                for result in results:
                    print(f"  - {result.movie['title']}")
                print("\nAnswer:")
                print(answer)
            case _:
                parser.print_help()
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
