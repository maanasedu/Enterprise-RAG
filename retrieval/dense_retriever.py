"""Dense retrieval: embed the query, search a vector store.

This is the only place that couples "a text query" to "an embedding", so
``HybridRetriever`` no longer embeds anything itself and the vector store
(FAISS today, Qdrant as a drop-in) never sees raw text.
"""
from typing import Dict, List

from retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        query_vec = self.embedder.embed([query])[0]
        return self.store.search(query_vec, top_k=top_k)
