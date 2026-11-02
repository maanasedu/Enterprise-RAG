import os
import faiss
import numpy as np
from typing import List, Dict
from ingestion.chunker import Chunk
from retrieval.base import VectorStore
from retrieval.identity import stable_point_id, chunk_identity_key

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
)


class FAISSStore(VectorStore):
    def __init__(self, dim: int = 1024):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

    def add(self, embeddings: List[List[float]], chunks: List[Chunk]):
        vecs = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.metadata.extend([
            {'text': c.text, 'metadata': c.metadata}
            for c in chunks
        ])

    def search(self, query_embedding: List[float], top_k: int = 50) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        q = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(q)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, top_k)
        return [
            {
                'text': self.metadata[i]['text'],
                'metadata': self.metadata[i]['metadata'],
                'score': float(scores[0][j]),
            }
            for j, i in enumerate(indices[0]) if i >= 0
        ]


class QdrantStore(VectorStore):
    def __init__(self, collection: str = 'documents'):
        os.makedirs('data/qdrant_storage', exist_ok=True)
        self.client = QdrantClient(path='data/qdrant_storage')
        self.collection = collection

    def create_collection(self, dim: int = 1024):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            self.client.delete_collection(collection_name=self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Collection '{self.collection}' created.")

    @staticmethod
    def _point_id_for(chunk: Chunk) -> str:
        """Stable, content-derived Qdrant point id for a chunk.

        Previously each call to add() assigned ids as `i + j`, where `i`
        restarted at 0 on every call (every upload, and every internal
        QDRANT_BATCH_SIZE batch within one upload). A second upload, or
        even the second internal batch of a large first upload, could
        therefore reuse ids already used by unrelated points and silently
        overwrite them in Qdrant.

        Using a UUID5 derived from the chunk's stable chunk_id (see
        ingestion.chunker.DocumentChunker) fixes this: every distinct
        chunk gets its own id, the same chunk always maps to the same id
        (so re-adding it is an idempotent upsert), and ids from different
        uploads cannot collide with each other.
        """
        chunk_id = chunk.metadata.get('chunk_id')
        if not chunk_id:
            # Legacy/callers that didn't run chunks through DocumentChunker
            # (and so have no chunk_id) still get a stable, collision-safe
            # id, derived from the full chunk text instead of a prefix.
            chunk_id = chunk_identity_key({'text': chunk.text, 'metadata': chunk.metadata})
        return stable_point_id(chunk_id)

    def add(self, embeddings: List[List[float]], chunks: List[Chunk]):
        BATCH_SIZE = 100
        total = len(chunks)
        for i in range(0, total, BATCH_SIZE):
            batch_emb = embeddings[i:i + BATCH_SIZE]
            batch_chunks = chunks[i:i + BATCH_SIZE]
            points = [
                PointStruct(
                    id=self._point_id_for(c),
                    vector=emb,
                    payload={'text': c.text, **c.metadata}
                )
                for emb, c in zip(batch_emb, batch_chunks)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            print(f'Uploaded {min(i + BATCH_SIZE, total)}/{total} to Qdrant')

    def search(self, query_embedding: List[float], top_k: int = 50) -> List[Dict]:
        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_embedding,
                limit=top_k,
            )
            return [
                {'text': r.payload['text'], 'metadata': r.payload, 'score': r.score}
                for r in results
            ]
        except Exception as e:
            print(f'Qdrant search error: {e}')
            return []
