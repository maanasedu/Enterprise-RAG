"""Hybrid retrieval: dense + BM25, fused with RRF, then reranked.

This class only *composes* the stages; each stage lives in its own module:

    dense   -> retrieval/dense_retriever.py  (DenseRetriever over a VectorStore)
    sparse  -> retrieval/bm25_retriever.py   (BM25Retriever)
    fusion  -> retrieval/fusion.py           (reciprocal_rank_fusion)
    rerank  -> retrieval/reranker.py         (rerank_pool / CohereReranker)

Public surface used by the API, AdaptiveRetriever, the evaluation code and
the tests is unchanged: ``retrieve``, ``retrieve_candidates``,
``rerank_candidates``, ``merge_candidate_pools``, ``_fuse_ranked_lists`` and
``_reciprocal_rank_fusion``, plus the mutable attributes ``dense``, ``bm25``,
``embedder`` and ``reranker``.

As a ``BaseRetriever``, ``search`` returns the fused, *unreranked* candidates
(``retrieve_candidates``); ``retrieve`` runs the full pipeline.
"""
from typing import List, Dict, Sequence

from retrieval.base import BaseRetriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.fusion import RRF_K, reciprocal_rank_fusion
from retrieval.reranker import CohereReranker, rerank_pool
from performance.recorder import BM25_RETRIEVAL, DENSE_RETRIEVAL, FUSION, RERANKING, stage


class HybridRetriever(BaseRetriever):
    def __init__(self, dense_store, bm25_retriever, embedder, use_reranker: bool = True):
        self.dense = dense_store
        self.bm25 = bm25_retriever
        self.embedder = embedder
        self.reranker = CohereReranker() if use_reranker else None

    @staticmethod
    def _fuse_ranked_lists(ranked_lists: Sequence[List[Dict]], k: int = RRF_K) -> List[Dict]:
        """RRF over any number of ranked lists (see retrieval.fusion)."""
        return reciprocal_rank_fusion(ranked_lists, k=k)

    def _reciprocal_rank_fusion(self, dense_results: List[Dict],
                                bm25_results: List[Dict], k: int = RRF_K) -> List[Dict]:
        return reciprocal_rank_fusion([dense_results, bm25_results], k=k)

    def merge_candidate_pools(self, pools: Sequence[List[Dict]],
                              limit: int = 50, k: int = RRF_K) -> List[Dict]:
        """Merge candidate pools from several queries (e.g. the sub-queries
        of a decomposed question) into one ranked pool of at most `limit`
        chunks, using the same RRF as the dense+BM25 fusion. A chunk found
        by several sub-queries is de-duplicated and ranks higher.
        """
        with stage(FUSION):
            return reciprocal_rank_fusion(pools, k=k)[:limit]

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        """BaseRetriever interface: fused candidates, not reranked."""
        return self.retrieve_candidates(query, top_k)

    def retrieve_candidates(self, query: str, top_k_retrieve: int = 50) -> List[Dict]:
        """Dense + BM25 retrieval fused with RRF, *without* reranking.

        Returns at most `top_k_retrieve` candidate chunks (best first), or
        [] when nothing was found.
        """
        # Built per call from the current attributes, so replacing
        # `hybrid.dense` / `hybrid.embedder` after construction still works.
        # D5-Lite: each stage is timed (no-op unless a request recorder is
        # active). Dense includes embedding the query; stages run sequentially.
        with stage(DENSE_RETRIEVAL):
            dense_results = DenseRetriever(self.dense, self.embedder).search(
                query, top_k=top_k_retrieve)
        with stage(BM25_RETRIEVAL):
            bm25_results = self.bm25.search(query, top_k=top_k_retrieve)
        with stage(FUSION):
            fused = self._reciprocal_rank_fusion(dense_results, bm25_results)
        return fused[:top_k_retrieve]

    def rerank_candidates(self, query: str, candidates_pool: List[Dict],
                          top_k_rerank: int = 5) -> List[Dict]:
        """Rerank an already-fused candidate pool against `query`."""
        with stage(RERANKING):
            return rerank_pool(self.reranker, query, candidates_pool, top_k_rerank)

    def retrieve(self, query: str, top_k_retrieve: int = 50,
                 top_k_rerank: int = 5) -> List[Dict]:
        candidates_pool = self.retrieve_candidates(query, top_k_retrieve)
        if not candidates_pool:
            return []
        return self.rerank_candidates(query, candidates_pool, top_k_rerank)
