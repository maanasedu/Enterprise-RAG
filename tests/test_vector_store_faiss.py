"""Regression tests for retrieval/vector_store.py FAISSStore — confirms
existing add/search behavior is unchanged by this pass."""
from retrieval.vector_store import FAISSStore
from ingestion.chunker import Chunk


def test_add_and_search_roundtrip():
    store = FAISSStore(dim=3)
    chunks = [
        Chunk(text='alpha chunk', metadata={'chunk_id': 'd1:recursive:0'}, strategy='recursive'),
        Chunk(text='beta chunk', metadata={'chunk_id': 'd1:recursive:1'}, strategy='recursive'),
    ]
    store.add([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], chunks)

    assert store.index.ntotal == 2
    assert len(store.metadata) == 2

    results = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]['text'] == 'alpha chunk'
    assert results[0]['metadata']['chunk_id'] == 'd1:recursive:0'


def test_search_on_empty_index_returns_empty_list():
    store = FAISSStore(dim=3)
    assert store.search([1.0, 0.0, 0.0], top_k=5) == []


def test_search_top_k_capped_at_ntotal():
    store = FAISSStore(dim=3)
    store.add([[1.0, 0.0, 0.0]], [Chunk(text='only chunk', metadata={'chunk_id': 'd1:recursive:0'}, strategy='recursive')])
    results = store.search([1.0, 0.0, 0.0], top_k=50)
    assert len(results) == 1
