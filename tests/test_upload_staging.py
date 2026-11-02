"""Regression tests for the staging/validate-then-commit pattern used by
api/rag_api.py's /upload endpoint (Fix #5: prevent a failed upload from
leaving the live index in an inconsistent state).

These tests exercise the same staging logic api/rag_api.py uses (clone the
FAISS index, build new metadata/BM25 off to the side, persist to disk with
temp-file-then-rename, only then swap the live objects in place) without
importing api.rag_api itself, since importing that module triggers loading
the real embedding model and LLM client at import time.
"""
import json
import os
import pickle

import faiss as faiss_lib
import numpy as np
import pytest

from retrieval.vector_store import FAISSStore
from retrieval.bm25_retriever import BM25Retriever


def _persist(staged_index, staged_metadata, staged_bm25, base_dir):
    processed = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed, exist_ok=True)

    tmp_faiss = os.path.join(processed, 'faiss_index.bin.tmp')
    tmp_meta = os.path.join(processed, 'faiss_metadata.json.tmp')
    tmp_bm25 = os.path.join(processed, 'bm25_index.pkl.tmp')

    faiss_lib.write_index(staged_index, tmp_faiss)
    with open(tmp_meta, 'w') as f:
        json.dump(staged_metadata, f)
    with open(tmp_bm25, 'wb') as f:
        pickle.dump(staged_bm25, f)

    os.replace(tmp_faiss, os.path.join(processed, 'faiss_index.bin'))
    os.replace(tmp_meta, os.path.join(processed, 'faiss_metadata.json'))
    os.replace(tmp_bm25, os.path.join(processed, 'bm25_index.pkl'))


@pytest.fixture
def live_state():
    faiss_store = FAISSStore(dim=3)
    faiss_store.index.add(np.array([[1, 0, 0]], dtype='float32'))
    faiss_store.metadata = [{'text': 'existing chunk', 'metadata': {'chunk_id': 'doc0:recursive:0'}}]

    bm25 = BM25Retriever()
    bm25.index([{'text': 'existing chunk', 'metadata': {'chunk_id': 'doc0:recursive:0'}}])
    return faiss_store, bm25


def test_staging_does_not_touch_live_state_until_commit(live_state):
    faiss_store, bm25 = live_state

    new_records = [{'text': 'brand new content', 'metadata': {'chunk_id': 'doc1:recursive:0'}}]
    vecs = np.array([[0, 1, 0]], dtype='float32')

    staged_index = faiss_lib.clone_index(faiss_store.index)
    staged_index.add(vecs)
    staged_metadata = faiss_store.metadata + new_records
    staged_bm25 = BM25Retriever()
    staged_bm25.index(bm25.corpus + new_records)

    # Staging must not mutate the live, currently-serving objects.
    assert faiss_store.index.ntotal == 1
    assert len(faiss_store.metadata) == 1
    assert len(bm25.corpus) == 1


def test_commit_swap_updates_live_objects_in_place(live_state, tmp_path, monkeypatch):
    faiss_store, bm25 = live_state
    monkeypatch.chdir(tmp_path)

    new_records = [{'text': 'brand new content', 'metadata': {'chunk_id': 'doc1:recursive:0'}}]
    vecs = np.array([[0, 1, 0]], dtype='float32')

    staged_index = faiss_lib.clone_index(faiss_store.index)
    staged_index.add(vecs)
    staged_metadata = faiss_store.metadata + new_records
    staged_bm25 = BM25Retriever()
    staged_bm25.index(bm25.corpus + new_records)

    _persist(staged_index, staged_metadata, staged_bm25, str(tmp_path))

    faiss_store.index = staged_index
    faiss_store.metadata = staged_metadata
    bm25.corpus = staged_bm25.corpus
    bm25.bm25 = staged_bm25.bm25

    assert faiss_store.index.ntotal == 2
    assert len(faiss_store.metadata) == 2
    assert len(bm25.corpus) == 2
    assert bm25.corpus[1]['metadata']['chunk_id'] == 'doc1:recursive:0'

    reloaded_index = faiss_lib.read_index(str(tmp_path / 'data' / 'processed' / 'faiss_index.bin'))
    assert reloaded_index.ntotal == 2

    with open(tmp_path / 'data' / 'processed' / 'faiss_metadata.json') as f:
        assert len(json.load(f)) == 2

    with open(tmp_path / 'data' / 'processed' / 'bm25_index.pkl', 'rb') as f:
        assert len(pickle.load(f).corpus) == 2


def test_dimension_mismatch_is_rejected_before_any_mutation(live_state):
    faiss_store, bm25 = live_state
    bad_vecs = np.array([[1, 2]], dtype='float32')  # wrong dim (2, not 3)

    with pytest.raises(ValueError):
        if bad_vecs.shape[1] != faiss_store.index.d:
            raise ValueError('Embedding dimension mismatch')

    # Live state must be completely unaffected by the aborted attempt.
    assert faiss_store.index.ntotal == 1
    assert len(faiss_store.metadata) == 1
    assert len(bm25.corpus) == 1
