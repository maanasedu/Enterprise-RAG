import re
from rank_bm25 import BM25Okapi
from typing import List, Dict

from retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self):
        self.bm25 = None
        self.corpus: List[Dict] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.sub(r'[^\w\s]', '', text.lower()).split()

    def index(self, chunks: List[Dict]):
        self.corpus = chunks
        tokenized = [self._tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        if not self.bm25:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            {
                'text': self.corpus[i]['text'],
                'metadata': self.corpus[i].get('metadata', {}),
                'bm25_score': float(scores[i])
            }
            for i in top_indices if scores[i] > 0
        ]