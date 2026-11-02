"""Tests for evaluation/retrieval_metrics.py. No API keys / network needed."""
import pytest

from evaluation.retrieval_metrics import (
    evaluate_retrieval,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


# ── recall_at_k ──────────────────────────────────────────────────────────

def test_recall_at_k_perfect():
    gold = ["a", "b"]
    retrieved = ["a", "b", "c", "d", "e"]
    assert recall_at_k(gold, retrieved, k=5) == 1.0


def test_recall_at_k_partial_with_multiple_gold_chunks():
    gold = ["a", "b", "c"]
    retrieved = ["x", "a", "y", "z", "w"]
    # only 'a' found in top 5 -> 1/3
    assert recall_at_k(gold, retrieved, k=5) == pytest.approx(1 / 3)


def test_recall_at_k_respects_k_cutoff():
    gold = ["a"]
    retrieved = ["x", "y", "a"]
    assert recall_at_k(gold, retrieved, k=2) == 0.0
    assert recall_at_k(gold, retrieved, k=3) == 1.0


def test_recall_at_k_zero_when_nothing_found():
    assert recall_at_k(["a", "b"], ["x", "y", "z"], k=10) == 0.0


def test_recall_at_k_requires_gold():
    with pytest.raises(ValueError):
        recall_at_k([], ["a", "b"], k=5)


# ── mrr ──────────────────────────────────────────────────────────────────

def test_mrr_first_position():
    assert mrr(["a"], ["a", "b", "c"]) == 1.0


def test_mrr_third_position():
    assert mrr(["c"], ["a", "b", "c"]) == pytest.approx(1 / 3)


def test_mrr_uses_first_relevant_hit_with_multiple_gold_chunks():
    # gold = {b, c}; first hit in ranked list is at position 2 (b)
    assert mrr(["b", "c"], ["a", "b", "c"]) == pytest.approx(1 / 2)


def test_mrr_zero_when_not_found():
    assert mrr(["z"], ["a", "b", "c"]) == 0.0


def test_mrr_requires_gold():
    with pytest.raises(ValueError):
        mrr([], ["a", "b"])


# ── ndcg_at_k ────────────────────────────────────────────────────────────

def test_ndcg_at_k_perfect_ranking_is_one():
    gold = ["a", "b"]
    retrieved = ["a", "b", "c"]
    assert ndcg_at_k(gold, retrieved, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_worse_ranking_is_less_than_one():
    gold = ["a", "b"]
    retrieved = ["c", "a", "b"]  # gold chunks present but not ideally ranked
    score = ndcg_at_k(gold, retrieved, k=3)
    assert 0.0 < score < 1.0


def test_ndcg_at_k_zero_when_nothing_found():
    assert ndcg_at_k(["a"], ["x", "y", "z"], k=3) == 0.0


def test_ndcg_at_k_handles_more_gold_than_k():
    # 3 gold chunks but k=1: ideal DCG uses min(k, |gold|) = 1 relevant slot.
    gold = ["a", "b", "c"]
    retrieved = ["a", "x", "y"]
    assert ndcg_at_k(gold, retrieved, k=1) == pytest.approx(1.0)


def test_ndcg_at_k_requires_gold():
    with pytest.raises(ValueError):
        ndcg_at_k([], ["a"], k=5)


# ── evaluate_retrieval ───────────────────────────────────────────────────

def test_evaluate_retrieval_averages_across_items_and_skips_no_gold_items():
    items = [
        {"question": "q1", "gold_chunk_ids": ["a"]},
        {"question": "q2", "gold_chunk_ids": ["z"]},
        {"question": "q3", "gold_chunk_ids": []},  # e.g. ambiguous/unanswerable -> skipped
    ]

    def fake_retrieve(question):
        return {"q1": ["a", "b"], "q2": ["b", "a"]}.get(question, [])

    result = evaluate_retrieval(items, fake_retrieve, k_values=(5,))
    assert result["n_evaluated"] == 2
    # q1: recall@5=1.0, mrr=1.0 ; q2: gold 'z' never found -> recall=0, mrr=0
    assert result["recall@5"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)


def test_evaluate_retrieval_handles_empty_input():
    result = evaluate_retrieval([], lambda q: [], k_values=(5,))
    assert result["n_evaluated"] == 0
