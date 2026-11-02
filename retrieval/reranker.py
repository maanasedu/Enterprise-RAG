import os
import time
import cohere
from typing import List, Dict, Optional

from retrieval.base import BaseReranker
from retrieval.identity import chunk_identity_key
from performance.recorder import RERANK_THROTTLE_WAIT, stage


class CohereReranker(BaseReranker):
    def __init__(self):
        self.client = cohere.Client(os.getenv('COHERE_API_KEY', ''))
        self.model = 'rerank-english-v3.0'
        self._last_call = 0
        self._min_interval = 6.5  # Respect 10 calls/minute free tier

    def _fallback(self, documents: List[str], top_n: int) -> List[Dict]:
        """Position-based fallback used when Cohere reranking is unavailable
        or fails. `relevance_score` here is a synthetic placeholder derived
        from the incoming rank order — NOT a model-produced relevance score.
        Every item is explicitly marked reranked=False so callers (and the
        API response) never report a degraded result as if it had actually
        been reranked.
        """
        return [
            {'text': d, 'relevance_score': 1.0 - (i * 0.1), 'index': i, 'reranked': False}
            for i, d in enumerate(documents[:top_n])
        ]

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict]:
        """Rerank `documents` for `query`.

        Every returned dict includes:
          - 'index': the position of this document in the *input*
            `documents` list, so callers can restore per-chunk metadata by
            direct lookup instead of matching on (truncated) text.
          - 'reranked': True only when Cohere's API actually scored the
            documents; False for the no-API-key and error fallback paths.
        """
        if not documents:
            return []
        if not os.getenv('COHERE_API_KEY', ''):
            # No key — return documents as-is with dummy scores
            return self._fallback(documents, top_n)
        # Rate limit protection
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            # D5-Lite: this sleep is part of reranking latency; time it
            # separately so it can be told apart from Cohere's own latency.
            with stage(RERANK_THROTTLE_WAIT):
                time.sleep(self._min_interval - elapsed)
        try:
            response = self.client.rerank(
                query=query,
                documents=documents[:50],  # Max 50 candidates
                top_n=top_n,
                model=self.model
            )
            self._last_call = time.time()
            return [
                {
                    'text': documents[r.index],
                    'relevance_score': r.relevance_score,
                    'index': r.index,
                    'reranked': True,
                }
                for r in response.results
            ]
        except Exception as e:
            print(f'Cohere rerank failed: {e} — returning top-{top_n} by position (not reranked)')
            return self._fallback(documents, top_n)



def rerank_pool(reranker: Optional[BaseReranker], query: str,
                candidates_pool: List[Dict], top_n: int = 5) -> List[Dict]:
    """Rerank an already-fused candidate pool against ``query``.

    This is the reranking *stage*, independent of any retriever (moved
    unchanged out of ``HybridRetriever.rerank_candidates``). ``reranker``
    may be any object with ``rerank(query, texts, top_n=...)``; ``None``
    means reranking is disabled for this pipeline.

    Metadata is restored by the exact position the reranker reports
    (``texts[idx]`` is guaranteed to be ``candidates_pool[idx]``) rather
    than by re-matching on truncated text, which could restore the wrong
    chunk's metadata whenever two chunks share a text prefix.
    """
    if not candidates_pool:
        return []

    if not reranker:
        # Reranker disabled entirely: keep fused order, mark not reranked.
        return [{**doc, 'reranked': False} for doc in candidates_pool[:top_n]]

    texts = [r['text'] for r in candidates_pool]
    reranked = reranker.rerank(query, texts, top_n=top_n)

    results = []
    for r in reranked:
        idx = r.get('index')
        source = (
            candidates_pool[idx]
            if idx is not None and 0 <= idx < len(candidates_pool)
            else None
        )
        metadata = source.get('metadata', {}) if source else {}
        results.append({
            'text': r['text'],
            'metadata': metadata,
            'chunk_id': chunk_identity_key(source) if source else None,
            'relevance_score': r.get('relevance_score'),
            # True only if Cohere's API actually scored this batch; False
            # for the no-key/error fallback so a degraded result is never
            # reported as a real reranking.
            'reranked': r.get('reranked', False),
        })
    return results
