"""Regression tests for retrieval/vector_store.py QdrantStore (Fix #2:
stable point ids so separate uploads cannot overwrite each other's chunks).

Uses qdrant-client's embedded/local mode against a temp directory, so no
external Qdrant server is required. Only one QdrantStore may hold an open
local-mode client on a given path at a time, matching how the real
/upload endpoint creates a fresh, short-lived QdrantStore per request.
"""
import gc
import pytest

pytest.importorskip('qdrant_client')

from retrieval.vector_store import QdrantStore
from ingestion.chunker import Chunk


@pytest.fixture
def qdrant_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_separate_uploads_do_not_collide(qdrant_dir):
    store1 = QdrantStore(collection='documents')
    store1.create_collection(dim=3)

    chunks1 = [
        Chunk(text='doc1 chunk0', metadata={'chunk_id': 'doc1:recursive:0'}, strategy='recursive'),
        Chunk(text='doc1 chunk1', metadata={'chunk_id': 'doc1:recursive:1'}, strategy='recursive'),
    ]
    store1.add([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], chunks1)
    del store1
    gc.collect()

    store2 = QdrantStore(collection='documents')
    chunks2 = [
        Chunk(text='doc2 chunk0', metadata={'chunk_id': 'doc2:recursive:0'}, strategy='recursive'),
        Chunk(text='doc2 chunk1', metadata={'chunk_id': 'doc2:recursive:1'}, strategy='recursive'),
    ]
    store2.add([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], chunks2)

    count = store2.client.count(collection_name='documents').count
    # Old `id = i + j` scheme reset per call, so the second upload would
    # have silently overwritten the first upload's two points, leaving 2.
    assert count == 4


def test_readding_same_chunk_is_idempotent_upsert(qdrant_dir):
    store = QdrantStore(collection='documents')
    store.create_collection(dim=3)

    chunk = Chunk(text='doc1 chunk0', metadata={'chunk_id': 'doc1:recursive:0'}, strategy='recursive')
    store.add([[1.0, 0.0, 0.0]], [chunk])
    store.add([[1.0, 0.0, 0.0]], [chunk])  # same chunk again

    count = store.client.count(collection_name='documents').count
    assert count == 1


def test_missing_chunk_id_falls_back_without_colliding(qdrant_dir):
    store = QdrantStore(collection='documents')
    store.create_collection(dim=3)

    chunk_a = Chunk(text='legacy chunk A text', metadata={}, strategy='recursive')
    chunk_b = Chunk(text='legacy chunk B text', metadata={}, strategy='recursive')
    store.add([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [chunk_a, chunk_b])

    count = store.client.count(collection_name='documents').count
    assert count == 2
