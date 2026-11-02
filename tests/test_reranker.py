"""Regression tests for retrieval/reranker.py (Fix #6: fallback results must
never be reported as if they were actually reranked by Cohere)."""
import os
import pytest

from retrieval.reranker import CohereReranker


@pytest.fixture(autouse=True)
def _no_cohere_key(monkeypatch):
    monkeypatch.delenv('COHERE_API_KEY', raising=False)


def test_no_api_key_returns_unreranked_fallback():
    reranker = CohereReranker()
    docs = ['doc0', 'doc1', 'doc2']
    out = reranker.rerank('query', docs, top_n=2)

    assert len(out) == 2
    assert all(o['reranked'] is False for o in out)
    # Preserves original order/index when falling back.
    assert [o['index'] for o in out] == [0, 1]
    assert [o['text'] for o in out] == ['doc0', 'doc1']


def test_exception_path_also_marked_unreranked(monkeypatch):
    monkeypatch.setenv('COHERE_API_KEY', 'fake-key-for-test')
    reranker = CohereReranker()

    def _raise(*args, **kwargs):
        raise RuntimeError('simulated Cohere outage')

    reranker.client.rerank = _raise
    reranker._min_interval = 0  # skip real sleep in test

    out = reranker.rerank('query', ['doc0', 'doc1'], top_n=2)
    assert all(o['reranked'] is False for o in out)


def test_successful_rerank_is_marked_true(monkeypatch):
    monkeypatch.setenv('COHERE_API_KEY', 'fake-key-for-test')
    reranker = CohereReranker()
    reranker._min_interval = 0

    class FakeResult:
        def __init__(self, index, score):
            self.index = index
            self.relevance_score = score

    class FakeResponse:
        results = [FakeResult(1, 0.9), FakeResult(0, 0.4)]

    reranker.client.rerank = lambda **kwargs: FakeResponse()

    out = reranker.rerank('query', ['doc0', 'doc1'], top_n=2)
    assert all(o['reranked'] is True for o in out)
    assert out[0]['text'] == 'doc1'
    assert out[0]['index'] == 1
    assert out[1]['text'] == 'doc0'
    assert out[1]['index'] == 0


def test_empty_documents_returns_empty():
    reranker = CohereReranker()
    assert reranker.rerank('query', [], top_n=5) == []
