"""Interfaces for the retrieval pipeline (Dimension 6: architecture cleanup).

    Query
      -> DenseRetriever   (BaseRetriever over a VectorStore: FAISS or Qdrant)
      -> BM25Retriever    (BaseRetriever, sparse)
      -> RRF fusion       (retrieval/fusion.py, pure function)
      -> Reranker         (BaseReranker: CohereReranker)
      -> AdaptiveRetriever (routing / evidence gate) -> generation

Each stage depends only on the shape of the one before it, so any single
stage can be replaced without touching the others:

  * a *retriever* turns a text query into a best-first list of chunk dicts
    ``{'text': ..., 'metadata': {...}, <score key>: ...}``;
  * a *vector store* turns an already-embedded query into the same chunk
    dicts;
  * a *reranker* re-orders candidate texts for a query.

These are abstract base classes for our own implementations. They are NOT
enforced at runtime on collaborators: ``HybridRetriever`` still accepts any
object with the right ``search`` / ``rerank`` method (the test suite relies
on that), so the ABCs document and type the contract rather than gate it.

Chunk identity is never defined here; it always comes from
``retrieval.identity`` (``chunk_identity_key`` / ``stable_point_id``).
"""
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseRetriever(ABC):
    """Text query in, ranked chunk dicts out (best first)."""

    @abstractmethod
    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        """Return at most ``top_k`` chunk dicts for ``query``, best first.

        Returns ``[]`` when nothing matches. Implementations must not
        rerank: reranking is a separate stage (``BaseReranker``).
        """


class VectorStore(ABC):
    """Dense index over pre-computed embeddings (FAISS, Qdrant)."""

    @abstractmethod
    def add(self, embeddings: List[List[float]], chunks: List) -> None:
        """Index ``chunks`` (``ingestion.chunker.Chunk``) with their embeddings."""

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 50) -> List[Dict]:
        """Return at most ``top_k`` ``{'text', 'metadata', 'score'}`` dicts, best first."""


class BaseReranker(ABC):
    """Re-orders candidate texts for one query."""

    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict]:
        """Return up to ``top_n`` dicts, best first, each containing:

        ``text``, ``relevance_score``, ``index`` (position in the INPUT
        ``documents`` list) and ``reranked`` (True only if a real model
        scored it; False for positional fallbacks).
        """
