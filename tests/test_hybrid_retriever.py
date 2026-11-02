"""Regression tests for retrieval/hybrid_retriever.py.

Covers Fix #1 (RRF fusion + metadata restoration no longer keyed on
text[:100], which could merge/misattribute distinct chunks sharing a text
prefix) and Fix #6 (reranked flag propagation, including the disabled-
reranker path preserving the current retrieval behavior).
"""
from retrieval.hybrid_retriever import HybridRetriever

SHARED_PREFIX = 'HEADER BOILERPLATE TEXT REPEATED ' * 5  # > 100 chars


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeDense:
    def __init__(self, results):
        self._results = results

    def search(self, query_vec, top_k):
        return self._results


class FakeBM25:
    def __init__(self, results):
        self._results = results

    def search(self, query, top_k):
        return self._results


def test_rrf_merges_true_duplicates_by_chunk_id_not_text_prefix():
    dense = [
        {'text': SHARED_PREFIX + 'chunk-A', 'metadata': {'chunk_id': 'docA:recursive:0'}, 'score': 0.9},
        {'text': SHARED_PREFIX + 'chunk-B', 'metadata': {'chunk_id': 'docB:recursive:0'}, 'score': 0.8},
    ]
    bm25 = [
        # Same chunk as dense's docB -> should merge into one entry.
        {'text': SHARED_PREFIX + 'chunk-B', 'metadata': {'chunk_id': 'docB:recursive:0'}, 'bm25_score': 5.0},
    ]
    hybrid = HybridRetriever(FakeDense(dense), FakeBM25(bm25), FakeEmbedder(), use_reranker=False)
    results = hybrid.retrieve('query', top_k_retrieve=10, top_k_rerank=5)

    # Old text[:100]-keyed scheme would have collapsed docA and docB into a
    # single entry (they share a >100-char prefix). Correct behavior keeps
    # them distinct (2 results), since they are genuinely different chunks.
    assert len(results) == 2
    ids = {r['metadata']['chunk_id'] for r in results}
    assert ids == {'docA:recursive:0', 'docB:recursive:0'}


def test_disabled_reranker_marks_results_as_not_reranked():
    dense = [{'text': 'doc text', 'metadata': {'chunk_id': 'doc1:recursive:0'}, 'score': 0.9}]
    hybrid = HybridRetriever(FakeDense(dense), FakeBM25([]), FakeEmbedder(), use_reranker=False)
    results = hybrid.retrieve('query', top_k_retrieve=10, top_k_rerank=5)
    assert len(results) == 1
    assert results[0]['reranked'] is False


def test_reranker_metadata_restored_by_index_not_text_prefix():
    # Two chunks sharing an identical >100-char text prefix — the old
    # text[:100] merge key used for metadata restoration would have
    # attributed the wrong metadata here.
    dense = [
        {'text': SHARED_PREFIX + 'chunk-A', 'metadata': {'chunk_id': 'docA:recursive:0'}, 'score': 0.9},
        {'text': SHARED_PREFIX + 'chunk-B', 'metadata': {'chunk_id': 'docB:recursive:0'}, 'score': 0.8},
    ]

    class FakeReranker:
        def rerank(self, query, documents, top_n):
            # Cohere reverses the order.
            return [
                {'text': documents[1], 'relevance_score': 0.99, 'index': 1, 'reranked': True},
                {'text': documents[0], 'relevance_score': 0.5, 'index': 0, 'reranked': True},
            ][:top_n]

    hybrid = HybridRetriever(FakeDense(dense), FakeBM25([]), FakeEmbedder(), use_reranker=False)
    hybrid.reranker = FakeReranker()

    results = hybrid.retrieve('query', top_k_retrieve=10, top_k_rerank=2)

    assert results[0]['metadata']['chunk_id'] == 'docB:recursive:0'
    assert results[1]['metadata']['chunk_id'] == 'docA:recursive:0'
    assert all(r['reranked'] for r in results)


def test_reranker_fallback_results_marked_not_reranked():
    dense = [{'text': 'doc text', 'metadata': {'chunk_id': 'doc1:recursive:0'}, 'score': 0.9}]

    class FakeFallbackReranker:
        def rerank(self, query, documents, top_n):
            return [
                {'text': d, 'relevance_score': 1.0 - i * 0.1, 'index': i, 'reranked': False}
                for i, d in enumerate(documents[:top_n])
            ]

    hybrid = HybridRetriever(FakeDense(dense), FakeBM25([]), FakeEmbedder(), use_reranker=False)
    hybrid.reranker = FakeFallbackReranker()

    results = hybrid.retrieve('query', top_k_retrieve=10, top_k_rerank=5)
    assert all(r['reranked'] is False for r in results)


def test_empty_results_return_empty_list():
    hybrid = HybridRetriever(FakeDense([]), FakeBM25([]), FakeEmbedder(), use_reranker=False)
    assert hybrid.retrieve('query') == []
