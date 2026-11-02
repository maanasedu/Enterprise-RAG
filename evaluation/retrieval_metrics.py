"""Retrieval-ground-truth metrics for the fixed benchmark.

Deliberately kept separate from evaluation/ragas_suite.py's generation
metrics (faithfulness / answer_relevancy / context_precision /
context_recall). Those score the LLM's *answer*; the functions here score
whether the *retriever* found the right evidence, using gold_chunk_ids
from the benchmark, not answer-text similarity. See docs/benchmark.md
"Retrieval vs generation metrics" for why this separation matters (it's
what lets you answer "did retrieval or generation improve?").

All functions take:
    gold_chunk_ids: Iterable[str]   -- ground-truth relevant chunk ids
    retrieved_chunk_ids: Sequence[str] -- ranked list, best first, as
        produced by chunk_identity_key() on retriever output

Multiple gold chunks are handled as follows (documented per Step 11):
    Recall@K = |gold ∩ retrieved[:K]| / |gold|
    MRR      = 1 / rank of the FIRST retrieved chunk that is in gold
               (0.0 if none of the top-ranked results are in gold)
    nDCG@K   = binary relevance (1 if a retrieved chunk is in gold, else 0);
               ideal ranking places all min(K, |gold|) gold chunks first.
               Graded relevance is not used since we have no principled
               basis for grading one gold chunk as more relevant than
               another (see docs/benchmark.md).
"""
import math
from typing import Dict, Iterable, List, Sequence


def recall_at_k(gold_chunk_ids: Iterable[str], retrieved_chunk_ids: Sequence[str], k: int) -> float:
    gold = set(gold_chunk_ids)
    if not gold:
        raise ValueError("recall_at_k requires at least one gold_chunk_id")
    retrieved_top_k = set(retrieved_chunk_ids[:k])
    return len(gold & retrieved_top_k) / len(gold)


def mrr(gold_chunk_ids: Iterable[str], retrieved_chunk_ids: Sequence[str]) -> float:
    gold = set(gold_chunk_ids)
    if not gold:
        raise ValueError("mrr requires at least one gold_chunk_id")
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(gold_chunk_ids: Iterable[str], retrieved_chunk_ids: Sequence[str], k: int) -> float:
    gold = set(gold_chunk_ids)
    if not gold:
        raise ValueError("ndcg_at_k requires at least one gold_chunk_id")

    def dcg(relevances: List[int]) -> float:
        return sum(
            rel / math.log2(i + 2)  # i is 0-indexed -> position i+1, +1 for log2(rank+1)
            for i, rel in enumerate(relevances)
        )

    retrieved_top_k = retrieved_chunk_ids[:k]
    relevances = [1 if cid in gold else 0 for cid in retrieved_top_k]
    actual = dcg(relevances)

    ideal_relevances = [1] * min(k, len(gold))
    ideal = dcg(ideal_relevances)
    if ideal == 0:
        return 0.0
    return actual / ideal


def evaluate_retrieval(
    benchmark_items: List[Dict],
    retrieve_fn,
    k_values: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    """Run `retrieve_fn(question) -> ranked list of chunk_id strings` over
    every benchmark item that HAS gold evidence (answerable items; items
    with evidence_status='ambiguous' or 'insufficient' are skipped, since
    Recall/MRR/nDCG are undefined without gold_chunk_ids), and average the
    metrics. Returns a flat dict, e.g. {'recall@5': 0.8, 'recall@10': 0.9,
    'mrr': 0.75, 'ndcg@5': 0.82, 'n_evaluated': 123}.
    """
    totals = {f"recall@{k}": 0.0 for k in k_values}
    totals.update({f"ndcg@{k}": 0.0 for k in k_values})
    totals["mrr"] = 0.0
    n = 0

    for item in benchmark_items:
        gold = item.get("gold_chunk_ids") or []
        if not gold:
            continue
        retrieved = retrieve_fn(item["question"])
        for k in k_values:
            totals[f"recall@{k}"] += recall_at_k(gold, retrieved, k)
            totals[f"ndcg@{k}"] += ndcg_at_k(gold, retrieved, k)
        totals["mrr"] += mrr(gold, retrieved)
        n += 1

    if n == 0:
        return {**{k: 0.0 for k in totals}, "n_evaluated": 0}

    result = {name: value / n for name, value in totals.items()}
    result["n_evaluated"] = n
    return result
