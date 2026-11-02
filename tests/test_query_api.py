"""Dimension 2: the real /query endpoint, end to end, with external services
faked (embedder, Cohere, Groq). Covers:

  10. existing API behavior unchanged for normal queries
   3. strong evidence -> generation proceeds
   5. weak evidence after retry -> structured refusal, no generation
   +  ADAPTIVE_ROUTING_ENABLED=false restores the exact legacy path

api.rag_api loads a real embedding model, Groq client and index files at
import time, so it is imported here with those pieces stubbed (the repo's
other tests avoid importing it for the same reason). The retrieval stack
under test — HybridRetriever, router, evaluator, reformulator, endpoint —
is the real code.
"""
import importlib
import sys
import types
from types import SimpleNamespace

import faiss
import pytest
from fastapi.testclient import TestClient

from retrieval.adaptive import AdaptiveConfig, REFUSAL_MESSAGE
from retrieval.evidence import EvidenceConfig
from tests.adaptive_helpers import ConstantScoreReranker, QueryKeyedBM25, real_hybrid

LOW_Q = 'What is the notice period for terminating the contract?'
LEGACY_KEYS = {'answer', 'sources', 'latency_ms', 'session_id', 'turn_number'}


class FakeGroq:
    """Stands in for RAGGenerator; records prompts sent for generation."""
    model = 'fake-model'

    def __init__(self, answer='The notice period is 30 days (Source 1).'):
        self.answer = answer
        self.prompts = []
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self._create)))

    def _create(self, **kwargs):
        self.prompts.append(kwargs['messages'])
        message = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Import a fresh api.rag_api with heavy deps stubbed; yield the module."""
    monkeypatch.chdir(tmp_path)          # no index files -> "not found" branches
    for var in ('ADAPTIVE_ROUTING_ENABLED', 'ADAPTIVE_REFORMULATION_MODE',
                'ADAPTIVE_MAX_SUBQUERIES', 'ADAPTIVE_LONG_QUERY_WORDS',
                'ADAPTIVE_EVIDENCE_TOP_K', 'ADAPTIVE_EVIDENCE_MIN_TOP_SCORE',
                'ADAPTIVE_EVIDENCE_MIN_MEAN_SCORE',
                'ADAPTIVE_EVIDENCE_FAIL_OPEN_WHEN_UNSCORED'):
        monkeypatch.delenv(var, raising=False)

    def missing_index(*args, **kwargs):
        raise FileNotFoundError('no index in test')

    monkeypatch.setattr(faiss, 'read_index', missing_index)

    fake_embedder_mod = types.ModuleType('ingestion.embedder')
    fake_embedder_mod.CachedEmbedder = lambda: SimpleNamespace(embed=lambda texts: [[0.0] * 3] * len(texts))
    fake_parser_mod = types.ModuleType('ingestion.parser')
    fake_parser_mod.DocumentParser = object
    monkeypatch.setitem(sys.modules, 'ingestion.embedder', fake_embedder_mod)
    monkeypatch.setitem(sys.modules, 'ingestion.parser', fake_parser_mod)

    import generation.llm_client as llm_client
    monkeypatch.setattr(llm_client, 'RAGGenerator', lambda provider='groq': FakeGroq())

    monkeypatch.delitem(sys.modules, 'api.rag_api', raising=False)
    module = importlib.import_module('api.rag_api')

    # A non-empty index so /query doesn't 503, and a clean session store.
    module.faiss_store = SimpleNamespace(index=SimpleNamespace(ntotal=1))
    module.conversation_store.clear()
    module.generator = FakeGroq()

    yield module
    sys.modules.pop('api.rag_api', None)


def wire(api, score):
    """Point the module at a real HybridRetriever backed by fakes."""
    reranker = ConstantScoreReranker(score)
    api.hybrid = real_hybrid(reranker, QueryKeyedBM25())
    return reranker


def post(api, question=LOW_Q, **extra):
    client = TestClient(api.app)
    return client.post('/query', json={'question': question, **extra})


# ── adaptive on, strong evidence: normal query still just works ──────────────
def test_normal_query_with_strong_evidence_generates_and_keeps_legacy_fields(api):
    api.adaptive_config = AdaptiveConfig(evidence=EvidenceConfig(min_top_score=0.3))
    wire(api, score=0.9)

    resp = post(api, top_k=3)
    assert resp.status_code == 200
    body = resp.json()

    assert LEGACY_KEYS <= set(body)                      # nothing legacy removed
    assert body['answer'] == 'The notice period is 30 days (Source 1).'
    assert body['session_id'] == 'default' and body['turn_number'] == 1
    assert len(body['sources']) == 1                     # QueryKeyedBM25 yields one chunk
    assert body['sources'][0]['reranked'] is True

    assert body['refused'] is False
    assert body['routing']['complexity'] == 'low' and body['routing']['path'] == 'direct'
    assert body['evidence']['sufficient'] is True and body['evidence']['retried'] is False
    assert 'refusal_reason' not in body
    assert len(api.generator.prompts) == 1               # generation happened, once
    assert [t['question'] for t in api.conversation_store['default']] == [LOW_Q]


def test_high_complexity_query_reports_decomposed_routing_and_generates(api):
    api.adaptive_config = AdaptiveConfig(evidence=EvidenceConfig(min_top_score=0.3))
    wire(api, score=0.9)
    body = post(api, 'What is attention? How does multi-head attention work?').json()
    assert body['routing']['path'] == 'decomposed'
    assert len(body['routing']['subqueries']) == 2
    assert body['refused'] is False and len(body['sources']) == 2


# ── weak evidence: structured refusal, no generation ─────────────────────────
def test_weak_evidence_after_retry_returns_structured_refusal_without_generating(api):
    api.adaptive_config = AdaptiveConfig(evidence=EvidenceConfig(min_top_score=0.3))
    reranker = wire(api, score=0.001)

    resp = post(api)
    assert resp.status_code == 200                        # a refusal is a valid answer, not an error
    body = resp.json()

    assert body['answer'] == REFUSAL_MESSAGE == 'Insufficient evidence to answer accurately.'
    assert body['refused'] is True
    assert body['refusal_reason'] == 'insufficient_evidence'
    assert body['sources'] == []                          # weak chunks are not presented as sources
    assert body['evidence']['retried'] is True
    assert [a['kind'] for a in body['evidence']['attempts']] == ['direct', 'reformulated']
    assert body['evidence']['sufficient'] is False
    assert api.generator.prompts == []                    # NEVER generate from weak evidence
    assert len(reranker.calls) == 2                       # bounded: initial + one retry
    assert api.conversation_store.get('default', []) == []  # refusals are not stored as history


def test_threshold_change_flips_the_same_query_from_refusal_to_answer(api):
    wire(api, score=0.2)
    api.adaptive_config = AdaptiveConfig(evidence=EvidenceConfig(min_top_score=0.5))
    assert post(api).json()['refused'] is True

    api.adaptive_config = AdaptiveConfig(evidence=EvidenceConfig(min_top_score=0.1))
    assert post(api).json()['refused'] is False


def test_retry_recovers_when_reformulated_query_finds_better_evidence(api):
    api.adaptive_config = AdaptiveConfig(evidence=EvidenceConfig(min_top_score=0.3))

    class ImprovesOnRetry(ConstantScoreReranker):
        def rerank(self, query, documents, top_n):
            self.score = 0.001 if len(self.calls) == 0 else 0.9
            return super().rerank(query, documents, top_n)

    api.hybrid = real_hybrid(ImprovesOnRetry(0.001), QueryKeyedBM25())
    body = post(api).json()
    assert body['refused'] is False and body['evidence']['retried'] is True
    assert len(api.generator.prompts) == 1


def test_unscored_evidence_still_answers_by_default_like_before(api):
    # Cohere unavailable -> reranker fallback marks results reranked=False.
    api.adaptive_config = AdaptiveConfig(evidence=EvidenceConfig(min_top_score=0.99))

    class Unavailable:
        def rerank(self, query, documents, top_n):
            return [{'text': d, 'relevance_score': 1.0 - i * 0.1, 'index': i, 'reranked': False}
                    for i, d in enumerate(documents[:top_n])]

    api.hybrid = real_hybrid(Unavailable(), QueryKeyedBM25())
    body = post(api).json()
    assert body['refused'] is False
    assert body['evidence']['evaluated'] is False
    assert len(api.generator.prompts) == 1


# ── 10. adaptive OFF == exact legacy behavior ────────────────────────────────
def test_disabled_adaptive_routing_restores_exact_legacy_response(api):
    api.adaptive_config = AdaptiveConfig(enabled=False)
    calls = []

    reranker = wire(api, score=0.001)         # would be refused if adaptive were on
    real_retrieve = api.hybrid.retrieve

    def spy(question, top_k_retrieve, top_k_rerank):
        calls.append((question, top_k_retrieve, top_k_rerank))
        return real_retrieve(question, top_k_retrieve=top_k_retrieve, top_k_rerank=top_k_rerank)

    api.hybrid.retrieve = spy
    body = post(api, top_k=4).json()

    assert set(body) == LEGACY_KEYS                       # byte-for-byte the old response shape
    assert calls == [(LOW_Q, 50, 4)]                      # the original retrieval call, untouched
    assert body['answer'] == 'The notice period is 30 days (Source 1).'
    assert len(reranker.calls) == 1                       # no retry, no evidence gate
    assert len(api.generator.prompts) == 1


# ── existing endpoint behavior that must not change ──────────────────────────
def test_empty_question_still_400(api):
    api.adaptive_config = AdaptiveConfig()
    assert post(api, '   ').status_code == 400


def test_empty_index_still_503(api):
    api.adaptive_config = AdaptiveConfig()
    api.faiss_store = SimpleNamespace(index=SimpleNamespace(ntotal=0))
    assert post(api).status_code == 503


def test_retrieval_errors_still_surface_as_500(api):
    api.adaptive_config = AdaptiveConfig()

    class Exploding:
        def retrieve(self, *a, **k):
            raise RuntimeError('index corrupted')

    api.hybrid = Exploding()
    resp = post(api)
    assert resp.status_code == 500 and 'index corrupted' in resp.json()['detail']


def test_multi_turn_history_still_recorded_for_answered_queries(api):
    api.adaptive_config = AdaptiveConfig(evidence=EvidenceConfig(min_top_score=0.3))
    wire(api, score=0.9)
    post(api, session_id='s1')
    second = post(api, 'And the renewal terms?', session_id='s1').json()
    assert second['turn_number'] == 2
    assert len(api.conversation_store['s1']) == 2
