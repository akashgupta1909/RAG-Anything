import pytest

from rag.evaluation.metrics import ndcg, percentile, reciprocal_rank, retrieval_scores


def test_retrieval_scores_dedupe_chunks_of_the_same_document() -> None:
    retrieved = ["https://a/x#s1", "https://a/x#s2", "https://a/y/", "https://a/z"]
    scores = retrieval_scores(retrieved, ["https://a/y", "https://a/q"], k=3)
    # ranked docs: x, y, z -> y found at rank 2
    assert scores["hit@3"] == 1.0
    assert scores["recall@3"] == 0.5
    assert scores["mrr"] == 0.5
    assert scores["precision@3"] == pytest.approx(1 / 3)


def test_reciprocal_rank_and_ndcg() -> None:
    assert reciprocal_rank(["a", "b"], {"c"}, 5) == 0.0
    assert ndcg(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert ndcg(["x", "a"], {"a"}, 2) == pytest.approx(1 / 1.5849625, rel=1e-6)


def test_percentile() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40
