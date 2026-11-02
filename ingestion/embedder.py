import os
import json
import hashlib
import redis
from sentence_transformers import SentenceTransformer
from typing import List


class CachedEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.dim = 1024
        try:
            self.redis_client = redis.from_url(
                os.getenv('REDIS_URL', 'redis://localhost:6379'),
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            self.cache_enabled = True
        except Exception:
            print('Redis unavailable — embedding cache disabled')
            self.cache_enabled = False
        self.cache_ttl = 86400  # 24 hours

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results = [None] * len(texts)
        uncached_indices, uncached_texts = [], []

        for i, text in enumerate(texts):
            if self.cache_enabled:
                key = f'emb:{hashlib.md5(text.encode()).hexdigest()}'
                try:
                    cached = self.redis_client.get(key)
                    if cached:
                        results[i] = json.loads(cached)
                        continue
                except Exception:
                    pass
            uncached_indices.append(i)
            uncached_texts.append(text)

        if uncached_texts:
            vecs = self.model.encode(
                uncached_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            for list_pos, original_idx in enumerate(uncached_indices):
                vec = vecs[list_pos].tolist()
                results[original_idx] = vec
                if self.cache_enabled:
                    try:
                        key = f'emb:{hashlib.md5(uncached_texts[list_pos].encode()).hexdigest()}'
                        self.redis_client.setex(key, self.cache_ttl, json.dumps(vec))
                    except Exception:
                        pass

        return results

    # LangChain-compatible methods for SemanticChunker
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        result = self.embed([text])
        return result[0] if result else []