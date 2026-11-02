"""Helpers for turning corpus chunks into gold evidence ids for the
benchmark, reusing the SAME identity logic the retrieval pipeline uses at
query time (retrieval/identity.py), so gold_chunk_ids computed here are
guaranteed to match the chunk_id/text-hash key that HybridRetriever, RRF
fusion, and reranker-metadata restoration actually produce for these
chunks.

Document identity note
-----------------------
`ingestion/parser.py` assigns `document_id` as a hash of the *raw file
bytes* at parse time. The committed `data/processed/faiss_metadata.json`
predates that field entirely (verified: 0/7492 chunks carry a
`document_id` or `chunk_id` — this is the "pre-built index loaded from
disk" legacy case `retrieval/identity.py` already documents and falls
back for). Since we don't have the original PDF bytes in this repo
snapshot, we cannot recompute the real content-hash document_id.

`document_id_for_chunk` therefore falls back to the chunk's `filename` as
a stable, unique-per-source-document surrogate — every distinct PDF in
this corpus does have a distinct filename. The fallback order
(`metadata.get('document_id') or metadata.get('filename')`) means that if
this corpus is ever re-ingested through `DocumentParser`/`DocumentChunker`
(which do populate real `document_id`/`chunk_id`), regenerating the
benchmark against the new metadata will automatically pick up the real
content-hash ids with no code change here.
"""
import json
from collections import defaultdict
from typing import Dict, List, Tuple

from retrieval.identity import chunk_identity_key


def document_id_for_chunk(metadata: Dict) -> str:
    doc_id = metadata.get("document_id")
    if doc_id:
        return str(doc_id)
    filename = metadata.get("filename")
    if not filename:
        raise ValueError(f"chunk metadata has neither document_id nor filename: {metadata!r}")
    return str(filename)


def load_corpus(faiss_metadata_path: str) -> List[Dict]:
    """Load data/processed/faiss_metadata.json into a list of
    {'text':..., 'metadata':...} dicts (its native format)."""
    with open(faiss_metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_indexes(corpus: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, str], Dict[str, List[Dict]]]:
    """Given the raw corpus (list of {'text','metadata'} dicts), return:

    - chunk_id_to_chunk: chunk_id -> the corpus entry (text + metadata)
    - chunk_id_to_document_id: chunk_id -> document_id (filename surrogate)
    - document_id_to_chunks: document_id -> list of corpus entries,
      ordered by chunk_index (for readable multi-chunk sampling)
    """
    chunk_id_to_chunk: Dict[str, Dict] = {}
    chunk_id_to_document_id: Dict[str, str] = {}
    document_id_to_chunks: Dict[str, List[Dict]] = defaultdict(list)

    for entry in corpus:
        cid = chunk_identity_key(entry)
        did = document_id_for_chunk(entry["metadata"])
        chunk_id_to_chunk[cid] = entry
        chunk_id_to_document_id[cid] = did
        document_id_to_chunks[did].append(entry)

    for did in document_id_to_chunks:
        document_id_to_chunks[did].sort(
            key=lambda e: e["metadata"].get("chunk_index", 0)
        )

    return chunk_id_to_chunk, chunk_id_to_document_id, dict(document_id_to_chunks)
