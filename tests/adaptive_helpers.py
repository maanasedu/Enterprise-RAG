"""Shared fakes for the Dimension 2 (adaptive routing) tests.

Nothing here touches the network, Cohere, Groq, or an embedding model.
"""
from typing import Dict, List, Optional

from retrieval.hybrid_retriever import HybridRetriever


def make_chunk(text: str, score: Optional[float], reranked: bool = True,
               chunk_id: Optional[str] = None) -> Dict:
    """A chunk shaped like HybridRetriever.rerank_candidates() output."""
    return {
        'text': text,
        'metadata': {'chunk_id': chunk_id or f'id:{text}'},
        'chunk_id': chunk_id or f'id:{text}',
        'relevance_score': score,
        'reranked': reranked,
    }


def strong_chunks(prefix: str = 'strong') -> List[Dict]:
    return [make_chunk(f'{prefix}-{i}', 0.9 - i * 0.1) for i in range(3)]


def weak_chunks(prefix: str = 'weak') -> List[Dict]:
    return [make_chunk(f'{prefix}-{i}', 0.004 - i * 0.001) for i in range(3)]


class FakeHybrid:
    """Records every call and serves scripted, already-reranked rounds.

    `rounds` is consumed one entry per *reranking* call (either `retrieve`
    or `rerank_candidates`); the last entry repeats once exhausted, so a
    runaway retry loop would show up as too many recorded calls, not as an
    IndexError.
    """

    def __init__(self, rounds: List[List[Dict]]):
        self.rounds = rounds
        self.retrieve_calls: List[Dict] = []
        self.candidate_calls: List[Dict] = []
        self.merge_calls: List[Dict] = []
        self.rerank_calls: List[Dict] = []

    def _next_round(self) -> List[Dict]:
        index = min(len(self.retrieve_calls) + len(self.rerank_calls) - 1,
                    len(self.rounds) - 1)
        return self.rounds[index]

    def retrieve(self, query, top_k_retrieve=50, top_k_rerank=5):
        self.retrieve_calls.append(
            {'query': query, 'top_k_retrieve': top_k_retrieve, 'top_k_rerank': top_k_rerank}
        )
        return self._next_round()

    def retrieve_candidates(self, query, top_k_retrieve=50):
        self.candidate_calls.append({'query': query, 'top_k_retrieve': top_k_retrieve})
        return [{'text': f'cand:{query}', 'metadata': {'chunk_id': f'cand:{query}'}}]

    def merge_candidate_pools(self, pools, limit=50, k=60):
        self.merge_calls.append({'pools': pools, 'limit': limit})
        return HybridRetriever._fuse_ranked_lists(pools, k=k)[:limit]

    def rerank_candidates(self, query, candidates_pool, top_k_rerank=5):
        self.rerank_calls.append(
            {'query': query, 'pool': candidates_pool, 'top_k_rerank': top_k_rerank}
        )
        return self._next_round()

    @property
    def reranking_rounds(self) -> int:
        return len(self.retrieve_calls) + len(self.rerank_calls)


class SpyReformulator:
    def __init__(self, queries=None, raises: Optional[Exception] = None):
        self.queries = queries if queries is not None else ['a brand new query']
        self.raises = raises
        self.calls: List[Dict] = []

    def reformulate(self, question, prior_queries, decision=None):
        self.calls.append(
            {'question': question, 'prior_queries': list(prior_queries), 'decision': decision}
        )
        if self.raises:
            raise self.raises
        return list(self.queries)


# ── fakes for driving the REAL HybridRetriever ───────────────────────────────
class FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeDense:
    def __init__(self, results=None):
        self._results = results or []

    def search(self, query_vec, top_k):
        return self._results


class QueryKeyedBM25:
    """Returns one chunk per query string, so each sub-query is
    distinguishable in the merged pool. Records the queries it saw."""

    def __init__(self):
        self.queries: List[str] = []

    def search(self, query, top_k):
        self.queries.append(query)
        return [{
            'text': f'chunk about {query}',
            'metadata': {'chunk_id': f'id:{query}'},
            'bm25_score': 1.0,
        }]


class ConstantScoreReranker:
    """Cohere stand-in: every document gets `score` (minus a tiny rank
    decrement), and every call is recorded."""

    def __init__(self, score: float):
        self.score = score
        self.calls: List[Dict] = []

    def rerank(self, query, documents, top_n):
        self.calls.append({'query': query, 'documents': list(documents), 'top_n': top_n})
        return [
            {'text': d, 'relevance_score': self.score - i * 0.001, 'index': i, 'reranked': True}
            for i, d in enumerate(documents[:top_n])
        ]


def real_hybrid(reranker, bm25=None) -> HybridRetriever:
    hybrid = HybridRetriever(FakeDense(), bm25 or QueryKeyedBM25(), FakeEmbedder(),
                             use_reranker=False)
    hybrid.reranker = reranker
    return hybrid
