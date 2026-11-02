"""Dimension 2 refactor safety: HybridRetriever.retrieve() was split into
retrieve_candidates() + rerank_candidates(). retrieve() must behave exactly
as before, and the new N-list fusion must reduce to the old 2-list RRF."""
from retrieval.hybrid_retriever import HybridRetriever
from tests.adaptive_helpers import ConstantScoreReranker, FakeDense, FakeEmbedder, real_hybrid


class ListBM25:
    def __init__(self, results):
        self._results = results

    def search(self, query, top_k):
        return self._results


def doc(name):
    return {'text': f'text {name}', 'metadata': {'chunk_id': name}}


def test_retrieve_equals_rerank_of_retrieve_candidates():
    dense = [doc('a'), doc('b'), doc('c')]
    bm25 = [doc('c'), doc('d')]
    hybrid = HybridRetriever(FakeDense(dense), ListBM25(bm25), FakeEmbedder(), use_reranker=False)
    hybrid.reranker = ConstantScoreReranker(0.5)

    direct = hybrid.retrieve('q', top_k_retrieve=10, top_k_rerank=3)
    pool = hybrid.retrieve_candidates('q', 10)
    composed = hybrid.rerank_candidates('q', pool, 3)
    assert direct == composed


def test_retrieve_candidates_respects_top_k_retrieve_and_does_not_rerank():
    reranker = ConstantScoreReranker(0.5)
    hybrid = real_hybrid(reranker)
    hybrid.dense = FakeDense([doc(str(i)) for i in range(10)])
    pool = hybrid.retrieve_candidates('q', top_k_retrieve=4)
    assert len(pool) == 4
    assert reranker.calls == []


def test_retrieve_returns_empty_without_calling_reranker_when_nothing_found():
    reranker = ConstantScoreReranker(0.5)
    hybrid = HybridRetriever(FakeDense([]), ListBM25([]), FakeEmbedder(), use_reranker=False)
    hybrid.reranker = reranker
    assert hybrid.retrieve('q') == []
    assert reranker.calls == []
    assert hybrid.rerank_candidates('q', []) == []


def test_two_list_rrf_unchanged_by_generalisation():
    hybrid = HybridRetriever(FakeDense(), ListBM25([]), FakeEmbedder(), use_reranker=False)
    dense = [doc('a'), doc('b'), doc('c')]
    bm25 = [doc('c'), doc('a')]
    fused = hybrid._reciprocal_rank_fusion(dense, bm25)
    # Hand-computed: a = 1/61 + 1/62, c = 1/63 + 1/61, b = 1/62  ->  a > c > b
    assert [d['metadata']['chunk_id'] for d in fused] == ['a', 'c', 'b']


def test_merge_candidate_pools_dedupes_ranks_shared_chunks_higher_and_honors_limit():
    hybrid = HybridRetriever(FakeDense(), ListBM25([]), FakeEmbedder(), use_reranker=False)
    pool_a = [doc('x'), doc('shared')]
    pool_b = [doc('y'), doc('shared')]
    merged = hybrid.merge_candidate_pools([pool_a, pool_b], limit=10)
    ids = [d['metadata']['chunk_id'] for d in merged]
    assert ids[0] == 'shared' and sorted(ids) == ['shared', 'x', 'y']
    assert len(hybrid.merge_candidate_pools([pool_a, pool_b], limit=2)) == 2
