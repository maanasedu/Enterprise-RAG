"""Regression tests for retrieval/identity.py.

Covers Fix #1 (stable document/chunk ids replacing text[:100] as identity)
and the id-derivation used by Fix #2 (Qdrant point ids).
"""
import pytest

from retrieval.identity import chunk_identity_key, stable_point_id


def test_prefers_stable_chunk_id():
    doc = {'text': 'hello world', 'metadata': {'chunk_id': 'doc1:recursive:0'}}
    assert chunk_identity_key(doc) == 'doc1:recursive:0'


def test_fallback_does_not_truncate_like_old_text_100_scheme():
    # Two distinct chunks that share an identical 100+ character prefix
    # (a repeated document header/boilerplate) must NOT collide, even
    # when chunk_id metadata is missing (e.g. a pre-built legacy index).
    shared_prefix = 'SECTION HEADER BOILERPLATE ' * 5  # > 100 chars
    doc_a = {'text': shared_prefix + 'unique tail A', 'metadata': {}}
    doc_b = {'text': shared_prefix + 'unique tail B', 'metadata': {}}
    assert chunk_identity_key(doc_a) != chunk_identity_key(doc_b)


def test_fallback_key_is_stable_for_identical_text():
    doc = {'text': 'some chunk text', 'metadata': {}}
    assert chunk_identity_key(doc) == chunk_identity_key(dict(doc))


def test_missing_metadata_key_is_handled():
    doc = {'text': 'no metadata key at all'}
    # Should not raise, and should be stable / non-empty.
    key = chunk_identity_key(doc)
    assert key and key == chunk_identity_key(doc)


def test_stable_point_id_deterministic_and_distinct():
    id1 = stable_point_id('doc1:recursive:0')
    id2 = stable_point_id('doc1:recursive:0')
    id3 = stable_point_id('doc1:recursive:1')
    assert id1 == id2
    assert id1 != id3


def test_stable_point_id_rejects_empty():
    with pytest.raises(ValueError):
        stable_point_id(None)
    with pytest.raises(ValueError):
        stable_point_id('')
