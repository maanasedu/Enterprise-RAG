"""Reciprocal Rank Fusion, as a pure function.

Extracted unchanged from ``HybridRetriever._fuse_ranked_lists`` so fusion has
one implementation, usable and testable without any retriever. It is used
both for dense+BM25 fusion and for merging the candidate pools of several
sub-queries (adaptive decomposition).
"""
from typing import Dict, List, Sequence

from retrieval.identity import chunk_identity_key

# Standard RRF constant. HybridRetriever._reciprocal_rank_fusion exposes the
# same default, and evaluation/retrieval_eval.py reports it from there.
RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: Sequence[List[Dict]], k: int = RRF_K) -> List[Dict]:
    """Fuse any number of ranked chunk lists with Reciprocal Rank Fusion.

    A chunk's score is the sum over lists of ``1 / (rank + 1 + k)``.

    Chunks are keyed by ``chunk_identity_key`` (stable ``chunk_id``, or a
    full-text hash fallback), never by a text prefix: a truncated-text key
    can collide between two different chunks (repeated headers/boilerplate)
    and silently merge their scores and drop one's metadata.

    Ties keep first-seen order (Python's sort is stable), so earlier lists
    win ties. The first-seen dict for each chunk is the one returned.
    """
    scores: dict = {}
    docs: dict = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = chunk_identity_key(doc)
            scores[key] = scores.get(key, 0) + 1 / (rank + 1 + k)
            docs.setdefault(key, doc)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[key] for key, _ in ordered]
