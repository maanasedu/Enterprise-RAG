"""Dimension 6 (architecture cleanup) tests.

Covers: the retrieval interfaces, each stage working on its own (dense,
BM25, RRF fusion, reranking), the composed hybrid pipeline over BOTH vector
backends (FAISS and Qdrant local mode), pipeline stage order, compatibility
with BM25 pickles written before the refactor, and the deterministic-ID
contract (golden values -- if one of these fails, chunk/point IDs changed and
the Dimension 1 benchmark's gold_chunk_ids are no longer valid).
"""
import base64
import gc
import pickle

import pytest

from ingestion.chunker import Chunk, DocumentChunker
from retrieval.base import BaseReranker, BaseRetriever, VectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.fusion import RRF_K, reciprocal_rank_fusion
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.identity import chunk_identity_key, stable_point_id
from retrieval.reranker import CohereReranker, rerank_pool
from retrieval.vector_store import FAISSStore, QdrantStore

SHARED_PREFIX = 'HEADER BOILERPLATE TEXT REPEATED ' * 5  # > 100 chars


def doc(cid, text=None, **extra):
    return {'text': text or f'text of {cid}', 'metadata': {'chunk_id': cid}, **extra}


class KeywordEmbedder:
    """Deterministic 3-d embedder: axis 0 = 'alpha', 1 = 'beta', 2 = 'gamma'."""
    def embed(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            v = [float('alpha' in t), float('beta' in t), float('gamma' in t)]
            out.append(v if any(v) else [0.1, 0.1, 0.1])
        return out


class ReverseReranker(BaseReranker):
    """Reorders deterministically (reverse) and reports real positions."""
    def rerank(self, query, documents, top_n=5):
        order = list(range(len(documents)))[::-1][:top_n]
        return [{'text': documents[i], 'relevance_score': 1.0 - 0.1 * n,
                 'index': i, 'reranked': True} for n, i in enumerate(order)]


def make_chunks():
    return [
        Chunk('alpha only', {'chunk_id': 'd1:recursive:0'}, 'recursive'),
        Chunk('beta only', {'chunk_id': 'd1:recursive:1'}, 'recursive'),
        Chunk('gamma only', {'chunk_id': 'd1:recursive:2'}, 'recursive'),
        Chunk('alpha beta mixed', {'chunk_id': 'd1:recursive:3'}, 'recursive'),
        # Neutral filler: with only 4 chunks a term found in 2 of them has a
        # BM25 IDF of exactly 0 (and BM25Retriever drops score <= 0).
        Chunk('delta filler one', {'chunk_id': 'd1:recursive:4'}, 'recursive'),
        Chunk('delta filler two', {'chunk_id': 'd1:recursive:5'}, 'recursive'),
    ]


def make_bm25(chunks):
    bm25 = BM25Retriever()
    bm25.index([{'text': c.text, 'metadata': c.metadata} for c in chunks])
    return bm25


# ── interfaces ───────────────────────────────────────────────────────────────
def test_components_implement_the_interfaces():
    assert issubclass(DenseRetriever, BaseRetriever)
    assert issubclass(BM25Retriever, BaseRetriever)
    assert issubclass(HybridRetriever, BaseRetriever)
    assert issubclass(CohereReranker, BaseReranker)
    assert issubclass(FAISSStore, VectorStore)
    assert issubclass(QdrantStore, VectorStore)


@pytest.mark.parametrize('abc', [BaseRetriever, BaseReranker, VectorStore])
def test_interfaces_cannot_be_instantiated(abc):
    with pytest.raises(TypeError):
        abc()


# ── each stage on its own ────────────────────────────────────────────────────
def test_dense_retriever_embeds_query_and_searches_store():
    chunks = make_chunks()
    store = FAISSStore(dim=3)
    emb = KeywordEmbedder()
    store.add(emb.embed([c.text for c in chunks]), chunks)

    results = DenseRetriever(store, emb).search('gamma please', top_k=2)

    assert len(results) == 2
    assert results[0]['metadata']['chunk_id'] == 'd1:recursive:2'
    assert 'score' in results[0]


def test_dense_retriever_on_empty_store_returns_empty():
    assert DenseRetriever(FAISSStore(dim=3), KeywordEmbedder()).search('alpha') == []


def test_bm25_works_through_base_interface():
    retriever: BaseRetriever = make_bm25(make_chunks())
    results = retriever.search('beta', top_k=10)
    ids = [r['metadata']['chunk_id'] for r in results]
    assert set(ids) == {'d1:recursive:1', 'd1:recursive:3'}
    assert all(r['bm25_score'] > 0 for r in results)
    assert BM25Retriever().search('beta') == []          # unindexed -> []


# Pickle written by the PRE-refactor BM25Retriever (the committed
# data/processed/bm25_index.pkl format): plain instance with `bm25`/`corpus`.
LEGACY_BM25_PICKLE = (
    'gASVbQIAAAAAAACMGHJldHJpZXZhbC5ibTI1X3JldHJpZXZlcpSMDUJNMjVSZXRyaWV2ZXKUk5QpgZR9lCiMBGJtMjWUjAlyYW5rX2JtMjWUjAlCTTI1T2thcGmUk5QpgZR9lCiMAmsxlEc/+AAAAAAAAIwBYpRHP+gAAAAAAACMB2Vwc2lsb26URz/QAAAAAAAAjAtjb3JwdXNfc2l6ZZRLA4wFYXZnZGyUR0AIAAAAAAAAjAlkb2NfZnJlcXOUXZQofZQojAVhbHBoYZRLAYwEYmV0YZRLAYwFZ2FtbWGUSwF1fZQojAVkZWx0YZRLAYwHZXBzaWxvbpRLAYwEemV0YZRLAXV9lCiMBWFscGhhlEsBjAVkZWx0YZRLAYwFdGhldGGUSwF1ZYwDaWRmlH2UKGgTRz+sBb4/uLUfaBRHP+BYrvqBFFJoFUc/4Fiu+oEUUmgXRz+sBb4/uLUfaBhHP+BYrvqBFFJoGUc/4Fiu+oEUUmgdRz/gWK76gRRSdYwHZG9jX2xlbpRdlChLA0sDSwNljAl0b2tlbml6ZXKUTowLYXZlcmFnZV9pZGaURz/MBb4/uLUfdWKMBmNvcnB1c5RdlCh9lCiMBHRleHSUjBBhbHBoYSBiZXRhIGdhbW1hlIwIbWV0YWRhdGGUfZSMCGNodW5rX2lklIwOZDE6cmVjdXJzaXZlOjCUc3V9lChoJ4wSZGVsdGEgZXBzaWxvbiB6ZXRhlGgpfZRoK4wOZDE6cmVjdXJzaXZlOjGUc3V9lChoJ4wRYWxwaGEgZGVsdGEgdGhldGGUaCl9lGgrjA5kMTpyZWN1cnNpdmU6MpRzdWV1Yi4='
)


def test_pre_refactor_bm25_pickle_still_loads_and_searches():
    bm25 = pickle.loads(base64.b64decode(LEGACY_BM25_PICKLE))
    assert isinstance(bm25, BM25Retriever) and isinstance(bm25, BaseRetriever)
    assert len(bm25.corpus) == 3
    results = bm25.search('epsilon', top_k=5)
    assert [r['metadata']['chunk_id'] for r in results] == ['d1:recursive:1']
    # and it can be re-indexed / re-pickled like the upload path does
    bm25.index(bm25.corpus + [{'text': 'new epsilon', 'metadata': {'chunk_id': 'd2:recursive:0'}}])
    assert len(pickle.loads(pickle.dumps(bm25)).corpus) == 4


# ── RRF fusion ───────────────────────────────────────────────────────────────
def test_rrf_score_math_and_order():
    a, b, c = doc('a'), doc('b'), doc('c')
    fused = reciprocal_rank_fusion([[a, b], [b, c]], k=60)
    # b: 1/62 + 1/61 ; a: 1/61 ; c: 1/62
    assert [d['metadata']['chunk_id'] for d in fused] == ['b', 'a', 'c']


def test_rrf_dedupes_by_chunk_id_not_text_prefix():
    a = doc('docA:recursive:0', SHARED_PREFIX + 'A')
    b = doc('docB:recursive:0', SHARED_PREFIX + 'B')
    fused = reciprocal_rank_fusion([[a, b], [dict(b, bm25_score=3.0)]])
    assert [d['metadata']['chunk_id'] for d in fused] == ['docB:recursive:0', 'docA:recursive:0']
    assert 'bm25_score' not in fused[0]            # first-seen dict is kept


def test_rrf_ties_keep_first_list_first_and_empty_inputs():
    fused = reciprocal_rank_fusion([[doc('x')], [doc('y')]])
    assert [d['metadata']['chunk_id'] for d in fused] == ['x', 'y']
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_hybrid_fusion_helpers_delegate_to_the_single_implementation():
    lists = [[doc('a'), doc('b')], [doc('b'), doc('c')]]
    expected = reciprocal_rank_fusion(lists)
    assert HybridRetriever._fuse_ranked_lists(lists) == expected
    h = HybridRetriever(None, None, None, use_reranker=False)
    assert h._reciprocal_rank_fusion(*lists) == expected
    assert h.merge_candidate_pools(lists, limit=2) == expected[:2]
    # evaluation/retrieval_eval.py reports this default as the protocol's rrf_k
    import inspect
    assert inspect.signature(HybridRetriever._reciprocal_rank_fusion).parameters['k'].default == RRF_K == 60


# ── reranking stage ──────────────────────────────────────────────────────────
def test_rerank_pool_restores_metadata_by_position_not_text():
    pool = [doc('docA:recursive:0', SHARED_PREFIX + 'A'), doc('docB:recursive:0', SHARED_PREFIX + 'B')]
    out = rerank_pool(ReverseReranker(), 'q', pool, top_n=2)
    assert [r['chunk_id'] for r in out] == ['docB:recursive:0', 'docA:recursive:0']
    assert [r['metadata']['chunk_id'] for r in out] == ['docB:recursive:0', 'docA:recursive:0']
    assert all(r['reranked'] is True for r in out)


def test_rerank_pool_disabled_and_empty():
    pool = [doc('a'), doc('b'), doc('c')]
    out = rerank_pool(None, 'q', pool, top_n=2)
    assert [d['metadata']['chunk_id'] for d in out] == ['a', 'b']
    assert all(d['reranked'] is False for d in out)
    assert rerank_pool(ReverseReranker(), 'q', [], top_n=5) == []


def test_cohere_reranker_without_key_falls_back_and_is_flagged(monkeypatch):
    monkeypatch.delenv('COHERE_API_KEY', raising=False)
    out = rerank_pool(CohereReranker(), 'q', [doc('a'), doc('b')], top_n=2)
    assert [r['chunk_id'] for r in out] == ['a', 'b']
    assert all(r['reranked'] is False for r in out)


# ── composed pipeline ────────────────────────────────────────────────────────
def build_hybrid(store, chunks, reranker=None):
    emb = KeywordEmbedder()
    store.add(emb.embed([c.text for c in chunks]), chunks)
    h = HybridRetriever(store, make_bm25(chunks), emb, use_reranker=False)
    h.reranker = reranker
    return h


def test_hybrid_over_faiss_fuses_dense_and_bm25():
    h = build_hybrid(FAISSStore(dim=3), make_chunks())
    ids = [r['metadata']['chunk_id'] for r in h.retrieve('alpha beta', top_k_retrieve=10, top_k_rerank=4)]
    assert ids[0] == 'd1:recursive:3'                       # best for dense AND bm25
    assert set(ids) <= {c.metadata['chunk_id'] for c in make_chunks()}
    assert len(ids) == len(set(ids))                        # RRF de-duplicated
    bm25_ids = {r['metadata']['chunk_id'] for r in h.bm25.search('alpha beta', top_k=10)}
    assert bm25_ids and bm25_ids <= set(ids)                # sparse hits made it into the fusion


def test_hybrid_over_qdrant_local_mode(tmp_path, monkeypatch):
    pytest.importorskip('qdrant_client')
    monkeypatch.chdir(tmp_path)
    store = QdrantStore(collection='documents')
    store.create_collection(dim=3)
    h = build_hybrid(store, make_chunks())
    ids = [r['metadata']['chunk_id'] for r in h.retrieve('alpha beta', top_k_retrieve=10, top_k_rerank=4)]
    assert ids[0] == 'd1:recursive:3'
    assert len(ids) == len(set(ids))
    del h, store
    gc.collect()


def test_faiss_and_qdrant_agree_on_dense_ranking(tmp_path, monkeypatch):
    pytest.importorskip('qdrant_client')
    monkeypatch.chdir(tmp_path)
    chunks, emb = make_chunks(), KeywordEmbedder()
    vectors = emb.embed([c.text for c in chunks])
    faiss_store = FAISSStore(dim=3)
    faiss_store.add(vectors, chunks)
    qdrant = QdrantStore(collection='documents')
    qdrant.create_collection(dim=3)
    qdrant.add(vectors, chunks)
    q = emb.embed(['alpha'])[0]
    f_ids = [r['metadata']['chunk_id'] for r in faiss_store.search(q, top_k=4)]
    q_ids = [r['metadata']['chunk_id'] for r in qdrant.search(q, top_k=4)]
    assert f_ids[0] == q_ids[0]
    assert set(f_ids) == set(q_ids)
    del qdrant
    gc.collect()


def test_pipeline_stage_order_dense_then_bm25_then_rerank():
    calls = []

    class Dense:
        def search(self, vec, top_k):
            calls.append('dense'); return [doc('a'), doc('b')]

    class Sparse:
        def search(self, q, top_k):
            calls.append('bm25'); return [doc('b'), doc('c')]

    class Rerank(BaseReranker):
        def rerank(self, q, documents, top_n=5):
            calls.append('rerank')
            return [{'text': documents[0], 'relevance_score': 0.5, 'index': 0, 'reranked': True}]

    h = HybridRetriever(Dense(), Sparse(), KeywordEmbedder(), use_reranker=False)
    h.reranker = Rerank()
    out = h.retrieve('q', top_k_retrieve=10, top_k_rerank=1)

    assert calls == ['dense', 'bm25', 'rerank']
    assert out[0]['metadata']['chunk_id'] == 'b'           # b is in both lists -> top after RRF


def test_search_is_fused_candidates_without_reranking():
    calls = []

    class Rerank(BaseReranker):
        def rerank(self, *a, **k):
            calls.append('rerank'); return []

    h = build_hybrid(FAISSStore(dim=3), make_chunks(), reranker=Rerank())
    assert h.search('alpha', top_k=3) == h.retrieve_candidates('alpha', 3)
    assert calls == []


def test_a_stage_can_be_replaced_with_a_custom_retriever():
    class Fixed(BaseRetriever):
        def search(self, query, top_k=50):
            return [doc('only')]

    h = HybridRetriever(FAISSStore(dim=3), Fixed(), KeywordEmbedder(), use_reranker=False)
    assert [d['metadata']['chunk_id'] for d in h.retrieve_candidates('anything')] == ['only']


# ── deterministic IDs (golden values) ────────────────────────────────────────
def test_stable_point_id_golden_values():
    assert stable_point_id('doc1:recursive:0') == '674a412d-9916-5224-a2a6-9b9ce3885bfb'
    assert stable_point_id('abc123def4567890:recursive:7') == '8b6ebdeb-f296-59ee-ad9e-4344cb7bfac3'
    assert stable_point_id('doc1:recursive:0') == stable_point_id('doc1:recursive:0')


def test_text_hash_fallback_golden_value():
    key = chunk_identity_key({'text': 'hello', 'metadata': {}})
    assert key == 'text-hash:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    assert chunk_identity_key(doc('x:recursive:0')) == 'x:recursive:0'


def test_chunk_ids_are_deterministic_and_keep_their_format():
    text = ('Alpha paragraph about retrieval.\n\n' * 40)
    meta = {'document_id': 'abc123def4567890', 'source': 's.pdf'}
    first = DocumentChunker().chunk(text, dict(meta), strategy='recursive', chunk_size=200, chunk_overlap=20)
    second = DocumentChunker().chunk(text, dict(meta), strategy='recursive', chunk_size=200, chunk_overlap=20)
    ids = [c.metadata['chunk_id'] for c in first]
    assert ids == [c.metadata['chunk_id'] for c in second]
    assert ids == [f'abc123def4567890:recursive:{i}' for i in range(len(first))]


def test_qdrant_point_id_still_derives_from_chunk_id():
    c = Chunk('x', {'chunk_id': 'doc1:recursive:0'}, 'recursive')
    assert QdrantStore._point_id_for(c) == '674a412d-9916-5224-a2a6-9b9ce3885bfb'
    legacy = Chunk('some text', {}, 'recursive')           # no chunk_id -> text-hash id
    assert QdrantStore._point_id_for(legacy) == stable_point_id(chunk_identity_key(
        {'text': 'some text', 'metadata': {}}))
