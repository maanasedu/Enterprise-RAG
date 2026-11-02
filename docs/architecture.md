# Retrieval Architecture (Dimension 6)

Dimension 6 is a structural refactor of `retrieval/`. It adds no features and
changes no behavior: API responses, ingestion, chunk/point IDs, the Dimension 1
benchmark, Dimension 2 adaptive routing and the security guardrails are all
unchanged, and the existing test suite passes without edits.

## Pipeline

```
Query
  │
  ▼  security input guard                       (security/, unchanged)
  ▼  QueryRouter  ── LOW: direct │ HIGH: sub-queries   (retrieval/router.py)
  │
  ▼  DenseRetriever ── embed query → VectorStore.search     (FAISS in production)
  ▼  BM25Retriever  ── sparse search over the same corpus
  ▼  reciprocal_rank_fusion  (RRF, k=60, keyed by chunk identity)
  ▼  rerank_pool ── CohereReranker  (reranked=False on fallback)
  │
  ▼  EvidenceEvaluator ── weak? one reformulated retry   (retrieval/adaptive.py)
  ▼  context sanitizer → generation → output guard       (security/, generation/)
```

`HybridRetriever` composes the middle four stages; `AdaptiveRetriever` composes
routing, evidence evaluation and the single retry around `HybridRetriever`.
Neither re-implements a stage.

## Components and responsibilities

| Module | Class / function | Responsibility | Does NOT |
|---|---|---|---|
| `retrieval/base.py` | `BaseRetriever`, `VectorStore`, `BaseReranker` | The contracts below | contain logic |
| `retrieval/vector_store.py` | `FAISSStore`, `QdrantStore` (`VectorStore`) | Store/search embeddings; Qdrant derives point ids from `chunk_id` | see raw text queries |
| `retrieval/dense_retriever.py` | `DenseRetriever` (`BaseRetriever`) | Embed the query, search a `VectorStore` | fuse or rerank |
| `retrieval/bm25_retriever.py` | `BM25Retriever` (`BaseRetriever`) | Sparse lexical search | embed or rerank |
| `retrieval/fusion.py` | `reciprocal_rank_fusion` | Merge ranked lists (dense+BM25, or sub-query pools) | retrieve |
| `retrieval/reranker.py` | `CohereReranker` (`BaseReranker`), `rerank_pool` | Re-order a fused pool; restore metadata by position | retrieve or fuse |
| `retrieval/hybrid_retriever.py` | `HybridRetriever` (`BaseRetriever`) | Wire dense → BM25 → RRF → rerank | implement any stage |
| `retrieval/router.py`, `evidence.py`, `reformulator.py`, `adaptive.py` | Dimension 2 | Route, gate on evidence, retry once | retrieve directly |
| `retrieval/identity.py` | `chunk_identity_key`, `stable_point_id` | The one definition of chunk identity | — |

Contracts:

* `BaseRetriever.search(query, top_k) -> [chunk dict]`, best first, no reranking.
  For `HybridRetriever`, `search` is the fused *unreranked* pool; `retrieve()`
  runs the full pipeline.
* `VectorStore.add(embeddings, chunks)` / `.search(query_embedding, top_k)`.
* `BaseReranker.rerank(query, texts, top_n)` returns dicts with `text`,
  `relevance_score`, `index` (position in the input) and `reranked`.

The ABCs document and type our own classes; they are not enforced on
collaborators. `HybridRetriever` accepts any object with a compatible
`search`/`rerank`, so a single stage can be swapped (a different sparse
retriever, a different reranker, a Qdrant store instead of FAISS) without
touching the others.

## Vector backends

Production serving still loads the pre-built **FAISS** index from
`data/processed/` (the Dimension 1 benchmark was measured on it, and the Qdrant
local store is not shipped with the repo). **Qdrant** implements the same
`VectorStore` interface and is already populated at ingestion and upload time;
passing a `QdrantStore` as the dense store of `HybridRetriever` works
(covered by `tests/test_retrieval_architecture.py`). Switching the serving
backend is a deployment decision, not part of this refactor.

## Deterministic identity

Unchanged and pinned by golden-value tests:

* `chunk_id = f"{document_id}:{strategy}:{chunk_index}"` (`ingestion/chunker.py`)
* Qdrant point id = `uuid5(NAMESPACE_URL, "enterprise-rag-chunk:" + chunk_id)`
* Fallback for chunks without a `chunk_id`: `"text-hash:" + sha256(full text)`

RRF keys, reranker metadata restoration, Qdrant upserts and the benchmark's
`gold_chunk_ids` all use these. Changing any of them invalidates the
benchmark.

## What changed in this refactor

* Added `base.py`, `fusion.py`, `dense_retriever.py`.
* RRF moved out of `HybridRetriever` into `fusion.py` (one implementation);
  `HybridRetriever._fuse_ranked_lists` / `_reciprocal_rank_fusion` /
  `merge_candidate_pools` delegate to it and keep their signatures.
* Rerank-and-restore-metadata moved into `reranker.rerank_pool`;
  `HybridRetriever.rerank_candidates` delegates.
* `HybridRetriever` no longer embeds queries itself (`DenseRetriever` does).
* `BM25Retriever`, `FAISSStore`, `QdrantStore`, `CohereReranker` now subclass
  the interfaces. Pre-refactor `bm25_index.pkl` files still load (tested).
* Removed `QdrantStore.search`'s unused `filter_condition` parameter and the
  unused `rebuild_index()` in `api/rag_api.py` (never called).

## Tests

`tests/test_retrieval_architecture.py` covers each stage on its own, the
composed pipeline over FAISS and Qdrant, stage order, legacy-pickle loading and
the ID golden values. Run everything with `python -m pytest -q`.
