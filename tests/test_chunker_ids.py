"""Regression tests for ingestion/chunker.py stable document_id/chunk_id
assignment (Fix #1)."""
from ingestion.chunker import DocumentChunker

LONG_TEXT = ('Paragraph one. ' * 40) + '\n\n' + ('Paragraph two. ' * 40)


def test_chunks_get_unique_stable_ids():
    chunker = DocumentChunker()
    meta = {'filename': 'doc1.pdf', 'document_id': 'abc123'}
    chunks = chunker.chunk(LONG_TEXT, meta, strategy='recursive', chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    ids = [c.metadata['chunk_id'] for c in chunks]
    assert len(set(ids)) == len(ids), 'chunk_ids must be unique within a document'
    assert all(c.metadata['document_id'] == 'abc123' for c in chunks)


def test_chunk_ids_stable_across_reingestion_of_same_document():
    chunker = DocumentChunker()
    meta = {'filename': 'doc1.pdf', 'document_id': 'abc123'}
    ids_first = [
        c.metadata['chunk_id']
        for c in chunker.chunk(LONG_TEXT, meta, strategy='recursive', chunk_size=200, chunk_overlap=20)
    ]
    ids_second = [
        c.metadata['chunk_id']
        for c in chunker.chunk(LONG_TEXT, meta, strategy='recursive', chunk_size=200, chunk_overlap=20)
    ]
    assert ids_first == ids_second


def test_document_id_falls_back_to_text_hash_when_not_supplied():
    chunker = DocumentChunker()
    meta = {'filename': 'doc2.pdf'}  # no document_id supplied
    chunks_a = chunker.chunk(LONG_TEXT, meta, strategy='recursive', chunk_size=200, chunk_overlap=20)
    chunks_b = chunker.chunk(LONG_TEXT, meta, strategy='recursive', chunk_size=200, chunk_overlap=20)
    assert [c.metadata['chunk_id'] for c in chunks_a] == [c.metadata['chunk_id'] for c in chunks_b]

    different_text_chunks = chunker.chunk(
        'Completely different content. ' * 30, meta, strategy='recursive', chunk_size=200, chunk_overlap=20
    )
    assert different_text_chunks[0].metadata['document_id'] != chunks_a[0].metadata['document_id']


def test_different_strategy_yields_different_chunk_ids():
    chunker = DocumentChunker()
    meta = {'filename': 'doc1.pdf', 'document_id': 'abc123'}
    recursive_chunks = chunker.chunk(LONG_TEXT, meta, strategy='recursive', chunk_size=200, chunk_overlap=20)
    sliding_chunks = chunker.chunk(LONG_TEXT, meta, strategy='sliding_window', chunk_size=50, chunk_overlap=10)
    assert recursive_chunks[0].metadata['chunk_id'] != sliding_chunks[0].metadata['chunk_id']
