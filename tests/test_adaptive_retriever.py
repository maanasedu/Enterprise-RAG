"""Dimension 2: adaptive routing orchestration (retrieval/adaptive.py).

Maps to the roadmap test list:
  1 low-complexity -> direct retrieval
  2 high-complexity -> decomposition path
  3 strong evidence -> generation may proceed (result not refused)
  4 weak evidence -> exactly one reformulation
  5 weak after retry -> refusal
  6 no infinite retries
  7 scores treated as ranking signals (see also test_evidence.py)
  8 configurable evidence threshold
  9 existing retrieval behavior unchanged
"""
import pytest

from retrieval.adaptive import (
    AdaptiveConfig,
    AdaptiveRetriever,
    RETRY_SKIPPED_NO_DISTINCT_QUERY,
    build_adaptive_retriever,
)
from retrieval.evidence import EvidenceConfig, EvidenceEvaluator
from retrieval.reformulator import HeuristicReformulator, LLMReformulator
from retrieval.router import QueryRouter
from tests.adaptive_helpers import (
    ConstantScoreReranker,
    FakeHybrid,
    QueryKeyedBM25,
    SpyReformulator,
    make_chunk,
    real_hybrid,
    strong_chunks,
    weak_chunks,
)

LOW_Q = 'What is the notice period for terminating the contract?'
HIGH_Q = 'What is attention? How does multi-head attention work?'
EVAL = EvidenceEvaluator(EvidenceConfig(min_top_score=0.3))


def make(hybrid, reformulator=None, evaluator=EVAL):
    return AdaptiveRetriever(hybrid, evaluator=evaluator,
                             reformulator=reformulator or SpyReformulator())


# ── 1. low complexity -> direct retrieval ────────────────────────────────────
def test_low_complexity_query_uses_the_existing_direct_retrieve_call_only():
    hybrid = FakeHybrid([strong_chunks()])
    result = make(hybrid).retrieve(LOW_Q, top_k_retrieve=50, top_k_rerank=4)

    assert hybrid.retrieve_calls == [
        {'query': LOW_Q, 'top_k_retrieve': 50, 'top_k_rerank': 4}
    ]
    assert hybrid.candidate_calls == [] and hybrid.rerank_calls == [] and hybrid.merge_calls == []
    assert not result.decision.is_high
    assert [a.kind for a in result.attempts] == ['direct']


# ── 2. high complexity -> decomposition path ─────────────────────────────────
def test_high_complexity_query_retrieves_per_subquery_merges_and_reranks_once_on_original():
    hybrid = FakeHybrid([strong_chunks()])
    result = make(hybrid).retrieve(HIGH_Q, top_k_retrieve=50, top_k_rerank=5)

    assert [c['query'] for c in hybrid.candidate_calls] == [
        'What is attention?', 'How does multi-head attention work?'
    ]
    assert len(hybrid.merge_calls) == 1
    assert len(hybrid.rerank_calls) == 1
    assert hybrid.rerank_calls[0]['query'] == HIGH_Q      # scored against what the user asked
    assert hybrid.retrieve_calls == []                    # not the direct path
    assert result.decision.is_high and result.decision.decomposed
    assert [a.kind for a in result.attempts] == ['decomposed']


def test_high_but_indivisible_query_falls_back_to_direct_retrieval():
    q = 'Summarize the main risks across the documents'      # HIGH, nothing to split
    hybrid = FakeHybrid([strong_chunks()])
    result = make(hybrid).retrieve(q)
    assert result.decision.is_high and not result.decision.decomposed
    assert len(hybrid.retrieve_calls) == 1 and hybrid.candidate_calls == []


# ── 3. strong evidence -> proceeds, no reformulation ─────────────────────────
def test_strong_evidence_is_not_refused_and_never_reformulates():
    hybrid = FakeHybrid([strong_chunks()])
    reformulator = SpyReformulator()
    result = make(hybrid, reformulator).retrieve(LOW_Q)

    assert result.sufficient and not result.refused and not result.retried
    assert result.chunks == strong_chunks()
    assert reformulator.calls == []
    assert hybrid.reranking_rounds == 1


# ── 4. weak evidence -> exactly one reformulation ────────────────────────────
def test_weak_then_strong_triggers_exactly_one_reformulated_retrieval():
    hybrid = FakeHybrid([weak_chunks(), strong_chunks('retry')])
    reformulator = SpyReformulator(['contract termination notice'])
    result = make(hybrid, reformulator).retrieve(LOW_Q)

    assert len(reformulator.calls) == 1
    assert reformulator.calls[0]['question'] == LOW_Q
    assert LOW_Q in reformulator.calls[0]['prior_queries']
    assert [c['query'] for c in hybrid.candidate_calls] == ['contract termination notice']
    # The retry is reranked against the ORIGINAL question, not the rewrite.
    assert hybrid.rerank_calls[-1]['query'] == LOW_Q
    assert result.retried and result.sufficient and not result.refused
    assert [a.kind for a in result.attempts] == ['direct', 'reformulated']
    assert result.chunks == strong_chunks('retry')       # the retry's evidence is what is returned


def test_high_complexity_weak_evidence_also_reformulates_once():
    hybrid = FakeHybrid([weak_chunks(), strong_chunks('retry')])
    reformulator = SpyReformulator(['multi head attention mechanism'])
    result = make(hybrid, reformulator).retrieve(HIGH_Q)
    assert [a.kind for a in result.attempts] == ['decomposed', 'reformulated']
    assert len(reformulator.calls) == 1 and result.sufficient


# ── 5. weak after retry -> refusal ───────────────────────────────────────────
def test_weak_after_retry_is_refused():
    hybrid = FakeHybrid([weak_chunks(), weak_chunks('retry')])
    result = make(hybrid).retrieve(LOW_Q)

    assert result.refused and not result.sufficient
    assert result.retried
    assert result.verdict.reason == 'top_score_below_threshold'
    ev = result.evidence_dict()
    assert ev['sufficient'] is False and ev['retried'] is True
    assert [a['kind'] for a in ev['attempts']] == ['direct', 'reformulated']


def test_empty_retrieval_is_weak_evidence_and_refused_after_one_retry():
    hybrid = FakeHybrid([[]])
    result = make(hybrid).retrieve(LOW_Q)
    assert result.refused and result.verdict.reason == 'no_evidence'
    assert hybrid.reranking_rounds == 2


# ── 6. no infinite retries ───────────────────────────────────────────────────
def test_retry_is_bounded_even_if_reformulator_always_has_new_queries():
    class EndlessReformulator:
        n = 0
        def reformulate(self, question, prior_queries, decision=None):
            EndlessReformulator.n += 1
            return [f'endless variant {EndlessReformulator.n}']

    hybrid = FakeHybrid([weak_chunks()])           # weak forever
    result = make(hybrid, EndlessReformulator()).retrieve(LOW_Q)

    assert result.refused
    assert EndlessReformulator.n == 1              # reformulator consulted once
    assert hybrid.reranking_rounds == 2            # 1 initial + at most 1 retry (== 2 Cohere calls)
    assert len(result.attempts) == 2


def test_retry_bound_holds_for_decomposed_queries_and_across_calls():
    hybrid = FakeHybrid([weak_chunks()])
    retriever = make(hybrid)
    for _ in range(3):
        before = hybrid.reranking_rounds
        result = retriever.retrieve(HIGH_Q)
        assert result.refused and hybrid.reranking_rounds - before == 2


def test_retry_is_skipped_when_reformulator_offers_nothing_new():
    hybrid = FakeHybrid([weak_chunks()])
    result = make(hybrid, SpyReformulator([])).retrieve(LOW_Q)
    assert result.refused and not result.retried
    assert result.retry_skipped_reason == RETRY_SKIPPED_NO_DISTINCT_QUERY
    assert hybrid.reranking_rounds == 1            # no pointless identical re-retrieval


def test_retry_is_skipped_when_reformulator_only_repeats_prior_queries():
    hybrid = FakeHybrid([weak_chunks()])
    result = make(hybrid, SpyReformulator([LOW_Q.lower(), LOW_Q])).retrieve(LOW_Q)
    assert result.retry_skipped_reason == RETRY_SKIPPED_NO_DISTINCT_QUERY
    assert hybrid.reranking_rounds == 1


def test_reformulator_failure_skips_retry_instead_of_crashing():
    hybrid = FakeHybrid([weak_chunks()])
    result = make(hybrid, SpyReformulator(raises=RuntimeError('boom'))).retrieve(LOW_Q)
    assert result.refused and hybrid.reranking_rounds == 1


def test_unscoreable_evidence_is_not_retried():
    # Cohere down: chunks carry synthetic scores; a retry could not fix that.
    unscored = [make_chunk('a', 1.0, reranked=False), make_chunk('b', 0.9, reranked=False)]
    reformulator = SpyReformulator()
    strict = EvidenceEvaluator(EvidenceConfig(fail_open_when_unscored=False))
    result = make(FakeHybrid([unscored]), reformulator, strict).retrieve(LOW_Q)
    assert result.refused and reformulator.calls == []


def test_unscored_evidence_passes_through_by_default_matching_legacy_behavior():
    unscored = [make_chunk('a', 1.0, reranked=False)]
    result = make(FakeHybrid([unscored]), evaluator=EvidenceEvaluator()).retrieve(LOW_Q)
    assert result.sufficient and result.verdict.evaluated is False and result.chunks == unscored


def test_router_failure_degrades_to_direct_retrieval():
    class BrokenRouter:
        config = QueryRouter().config
        def route(self, q):
            raise RuntimeError('router bug')

    hybrid = FakeHybrid([strong_chunks()])
    result = AdaptiveRetriever(hybrid, router=BrokenRouter(), evaluator=EVAL).retrieve(HIGH_Q)
    assert result.sufficient
    assert len(hybrid.retrieve_calls) == 1 and hybrid.candidate_calls == []
    assert result.decision.signals == ('router_error',)


# ── 8. configurable threshold, end to end ────────────────────────────────────
def test_same_retrieval_is_accepted_or_refused_purely_by_threshold_config():
    moderate = [make_chunk('m0', 0.4), make_chunk('m1', 0.3)]
    lenient = AdaptiveRetriever(FakeHybrid([moderate]),
                                evaluator=EvidenceEvaluator(EvidenceConfig(min_top_score=0.35)))
    strict = AdaptiveRetriever(FakeHybrid([moderate]), reformulator=SpyReformulator([]),
                               evaluator=EvidenceEvaluator(EvidenceConfig(min_top_score=0.45)))
    assert lenient.retrieve(LOW_Q).sufficient
    assert strict.retrieve(LOW_Q).refused


# ── 7. ranking signals, at the pipeline level ────────────────────────────────
def test_pipeline_works_on_non_probability_score_scales():
    logit_like = [make_chunk('a', 7.2), make_chunk('b', -1.4)]
    cfg = EvidenceConfig(min_top_score=3.0)
    assert AdaptiveRetriever(FakeHybrid([logit_like]),
                             evaluator=EvidenceEvaluator(cfg)).retrieve(LOW_Q).sufficient


# ── 9. integration with the REAL HybridRetriever ─────────────────────────────
def test_decomposition_merges_evidence_from_every_subquery_into_one_rerank():
    bm25 = QueryKeyedBM25()
    reranker = ConstantScoreReranker(0.9)
    result = AdaptiveRetriever(real_hybrid(reranker, bm25), evaluator=EVAL).retrieve(HIGH_Q)

    assert bm25.queries == ['What is attention?', 'How does multi-head attention work?']
    assert len(reranker.calls) == 1                         # ONE Cohere call, not one per sub-query
    assert reranker.calls[0]['query'] == HIGH_Q
    assert set(reranker.calls[0]['documents']) == {
        'chunk about What is attention?',
        'chunk about How does multi-head attention work?',
    }
    assert result.sufficient
    assert {c['metadata']['chunk_id'] for c in result.chunks} == {
        'id:What is attention?', 'id:How does multi-head attention work?'}


def test_low_complexity_result_is_identical_to_plain_hybrid_retrieve():
    bm25, reranker = QueryKeyedBM25(), ConstantScoreReranker(0.8)
    hybrid = real_hybrid(reranker, bm25)
    adaptive = AdaptiveRetriever(hybrid, evaluator=EVAL).retrieve(LOW_Q, 50, 5).chunks
    plain = hybrid.retrieve(LOW_Q, top_k_retrieve=50, top_k_rerank=5)
    assert adaptive == plain


def test_real_pipeline_makes_at_most_two_cohere_calls_when_evidence_stays_weak():
    reranker = ConstantScoreReranker(0.001)
    result = AdaptiveRetriever(real_hybrid(reranker), evaluator=EVAL).retrieve(LOW_Q)
    assert result.refused and len(reranker.calls) == 2
    assert reranker.calls[0]['query'] == reranker.calls[1]['query'] == LOW_Q   # both scored vs original


# ── config + builder ─────────────────────────────────────────────────────────
def test_adaptive_config_from_env():
    cfg = AdaptiveConfig.from_env({
        'ADAPTIVE_ROUTING_ENABLED': 'false',
        'ADAPTIVE_REFORMULATION_MODE': 'LLM',
        'ADAPTIVE_MAX_SUBQUERIES': '2',
        'ADAPTIVE_LONG_QUERY_WORDS': '20',
        'ADAPTIVE_EVIDENCE_MIN_TOP_SCORE': '0.25',
    })
    assert cfg.enabled is False
    assert cfg.reformulation_mode == 'llm'
    assert cfg.router.max_subqueries == 2 and cfg.router.long_query_words == 20
    assert cfg.evidence.min_top_score == 0.25


def test_adaptive_config_defaults_and_validation():
    cfg = AdaptiveConfig.from_env({})
    assert cfg.enabled is True and cfg.reformulation_mode == 'heuristic'
    with pytest.raises(ValueError):
        AdaptiveConfig.from_env({'ADAPTIVE_REFORMULATION_MODE': 'magic'})
    with pytest.raises(ValueError, match='ADAPTIVE_MAX_SUBQUERIES'):
        AdaptiveConfig.from_env({'ADAPTIVE_MAX_SUBQUERIES': 'four'})


def test_builder_uses_heuristic_reformulator_by_default_and_wires_config():
    cfg = AdaptiveConfig(evidence=EvidenceConfig(min_top_score=0.77))
    retriever = build_adaptive_retriever(FakeHybrid([]), cfg)
    assert isinstance(retriever.reformulator, HeuristicReformulator)
    assert retriever.evaluator.config.min_top_score == 0.77


def test_builder_uses_llm_reformulator_only_when_configured_and_available():
    class Gen:
        client, model = object(), 'some-model'

    cfg = AdaptiveConfig(reformulation_mode='llm')
    assert isinstance(build_adaptive_retriever(FakeHybrid([]), cfg, Gen()).reformulator,
                      LLMReformulator)
    # llm requested but no client -> safe fallback, not a crash
    assert isinstance(build_adaptive_retriever(FakeHybrid([]), cfg, None).reformulator,
                      HeuristicReformulator)
    # client available but mode is heuristic -> no LLM use
    assert isinstance(build_adaptive_retriever(FakeHybrid([]), AdaptiveConfig(), Gen()).reformulator,
                      HeuristicReformulator)
