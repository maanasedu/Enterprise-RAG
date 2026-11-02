"""Stable identity helpers for chunks, shared across the retrieval pipeline.

These replace the old `text[:100]` (or `text[:50]`) truncated-text keys that
were previously used both to merge duplicate chunks during Reciprocal Rank
Fusion and to restore per-chunk metadata after Cohere reranking. Truncated
text is not a safe identity key: two different chunks that happen to share
the same first 100 characters (a repeated document header, boilerplate,
a running title) would silently collide, causing one chunk's metadata (or
score contribution) to overwrite the other's.

`chunk_identity_key` prefers the stable `chunk_id` assigned at ingestion
time (see `ingestion.chunker.DocumentChunker`). For chunks indexed before
`chunk_id` existed (e.g. a pre-built FAISS/BM25 index loaded from disk),
it falls back to a hash of the *entire* chunk text rather than a prefix,
which is still collision-safe and requires no re-ingestion.
"""
import hashlib
import uuid
from typing import Dict, Optional


def chunk_identity_key(doc: Dict) -> str:
    """Return a stable identity key for a retrieved chunk dict.

    `doc` is expected to look like the dicts returned by FAISSStore.search,
    BM25Retriever.search, etc: {'text': ..., 'metadata': {...}, ...}.
    """
    metadata = doc.get('metadata') or {}
    chunk_id = metadata.get('chunk_id')
    if chunk_id:
        return str(chunk_id)
    text = doc.get('text', '') or ''
    return 'text-hash:' + hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()


def stable_point_id(chunk_id: Optional[str]) -> str:
    """Deterministic Qdrant point ID derived from a stable chunk_id.

    Qdrant point IDs must be an unsigned integer or a UUID. UUID5 (a
    deterministic, content-derived UUID) gives every distinct chunk_id its
    own point id, and gives the *same* chunk_id the *same* point id every
    time — so re-adding the same chunk overwrites (upserts) its own point
    instead of colliding with an unrelated chunk from a different upload.

    `chunk_id` should not normally be missing (DocumentChunker always sets
    one), but if it is, the caller is expected to pass a text-hash fallback
    (see chunk_identity_key) so this still never collides across chunks.
    """
    if not chunk_id:
        raise ValueError('stable_point_id requires a non-empty chunk_id')
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'enterprise-rag-chunk:{chunk_id}'))
