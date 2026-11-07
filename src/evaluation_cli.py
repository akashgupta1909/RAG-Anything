"""Command-line interface for evaluating search quality against a golden dataset."""

import argparse
import json
import sys
from typing import TypedDict, cast

from config import GOLDEN_DATASET_FILE, RRF_K
from lib.dataset import load_movies
from lib.hybrid_search import HybridSearch


class TestCase(TypedDict):
    query: str
    relevant_docs: list[str]


class GoldenDataset(TypedDict):
    test_cases: list[TestCase]


def _load_golden_dataset() -> list[TestCase]:
    with GOLDEN_DATASET_FILE.open(encoding="utf-8") as file:
        return cast(GoldenDataset, json.load(file))["test_cases"]


def precision_at_k(retrieved: list[str], relevant: list[str]) -> float:
    """Fraction of the retrieved titles that are in the ``relevant`` set."""
    if not retrieved:
        return 0.0
    relevant_titles = set(relevant)
    return sum(title in relevant_titles for title in retrieved) / len(retrieved)


def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    """Fraction of the ``relevant`` titles that appear among the retrieved ones."""
    if not relevant:
        return 0.0
    retrieved_titles = set(retrieved)
    return sum(title in retrieved_titles for title in relevant) / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of ``precision`` and ``recall`` (0 when both are 0)."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit: int = args.limit
    if limit <= 0:
        parser.error(f"--limit must be positive, got {limit}")

    try:
        test_cases = _load_golden_dataset()
        search = HybridSearch(load_movies())  # load the model and index only once
        print(f"k={limit}\n")
        for case in test_cases:
            results = search.rrf_search(case["query"], RRF_K, limit)
            retrieved = [result.movie["title"] for result in results]
            precision = precision_at_k(retrieved, case["relevant_docs"])
            recall = recall_at_k(retrieved, case["relevant_docs"])
            print(f"- Query: {case['query']}")
            print(f"  - Precision@{limit}: {precision:.4f}")
            print(f"  - Recall@{limit}: {recall:.4f}")
            print(f"  - F1 Score: {f1_score(precision, recall):.4f}")
            print(f"  - Retrieved: {', '.join(retrieved)}")
            print(f"  - Relevant: {', '.join(case['relevant_docs'])}")
            print()
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
