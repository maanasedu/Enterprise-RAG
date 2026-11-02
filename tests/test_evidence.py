"""Dimension 2: evidence evaluation (retrieval/evidence.py).

Covers requirement 7 (reranker scores are ranking signals, not
probabilities) and requirement 8 (configurable evidence threshold).
"""
import dataclasses
import inspect

import pytest

from retrieval import evidence
from retrieval.evidence import (
    DEFAULT_MIN_TOP_SCORE,
    EvidenceConfig,
    EvidenceEvaluator,
    REASON_MEAN_SCORE_BELOW,
    REASON_NO_EVIDENCE,
    REASON_OK,
    REASON_TOP_SCORE_BELOW,
    REASON_UNSCORED_ACCEPTED,
    REASON_UNSCORED_REFUSED,
)
from tests.adaptive_helpers import make_chunk, strong_chunks, weak_chunks


def chunks_with(*scores):
    return [make_chunk(f'c{i}', s) for i, s in enumerate(scores)]


# ── basic verdicts ───────────────────────────────────────────────────────────
def test_strong_evidence_is_sufficient():
    v = EvidenceEvaluator(EvidenceConfig(min_top_score=0.3)).evaluate(strong_chunks())
    assert v.sufficient and v.evaluated
    assert v.reason == REASON_OK
    assert v.top_score == pytest.approx(0.9)
    assert not v.retryable


def test_weak_evidence_is_insufficient_and_retryable():
    v = EvidenceEvaluator(EvidenceConfig(min_top_score=0.3)).evaluate(weak_chunks())
    assert not v.sufficient and v.evaluated
    assert v.reason == REASON_TOP_SCORE_BELOW
    assert v.retryable


def test_no_chunks_is_insufficient_and_retryable():
    v = EvidenceEvaluator().evaluate([])
    assert not v.sufficient
    assert v.reason == REASON_NO_EVIDENCE
    assert v.retryable
    assert v.top_score is None


def test_threshold_boundary_is_inclusive():
    cfg = EvidenceConfig(min_top_score=0.5)
    assert EvidenceEvaluator(cfg).evaluate(chunks_with(0.5)).sufficient
    assert not EvidenceEvaluator(cfg).evaluate(chunks_with(0.4999)).sufficient


def test_input_order_does_not_matter():
    cfg = EvidenceConfig(top_k=2, min_top_score=0.5)
    ascending = EvidenceEvaluator(cfg).evaluate(chunks_with(0.1, 0.2, 0.9))
    descending = EvidenceEvaluator(cfg).evaluate(chunks_with(0.9, 0.2, 0.1))
    assert ascending.top_score == descending.top_score == 0.9
    assert ascending.scores_considered == descending.scores_considered == (0.9, 0.2)


# ── requirement 8: configurable threshold ────────────────────────────────────
def test_same_evidence_flips_verdict_when_threshold_changes():
    chunks = chunks_with(0.4, 0.3, 0.2)
    assert EvidenceEvaluator(EvidenceConfig(min_top_score=0.35)).evaluate(chunks).sufficient
    assert not EvidenceEvaluator(EvidenceConfig(min_top_score=0.45)).evaluate(chunks).sufficient


def test_top_k_controls_how_many_scores_are_inspected():
    chunks = chunks_with(0.9, 0.8, 0.0)
    cfg = dict(min_top_score=0.0, min_mean_score=0.6)  # mean@2=0.85, mean@3=0.567
    assert EvidenceEvaluator(EvidenceConfig(top_k=2, **cfg)).evaluate(chunks).sufficient
    v = EvidenceEvaluator(EvidenceConfig(top_k=3, **cfg)).evaluate(chunks)
    assert not v.sufficient and v.reason == REASON_MEAN_SCORE_BELOW
    assert len(v.scores_considered) == 3


def test_mean_check_is_off_by_default():
    v = EvidenceEvaluator(EvidenceConfig(min_top_score=0.1)).evaluate(chunks_with(0.9, 0.0, 0.0))
    assert v.sufficient
    assert v.min_mean_score is None


def test_verdict_reports_the_thresholds_it_used():
    v = EvidenceEvaluator(EvidenceConfig(min_top_score=0.42, min_mean_score=0.1)).evaluate(
        strong_chunks())
    assert v.to_dict()['thresholds'] == {'min_top_score': 0.42, 'min_mean_score': 0.1}


def test_config_from_env():
    cfg = EvidenceConfig.from_env({
        'ADAPTIVE_EVIDENCE_TOP_K': '2',
        'ADAPTIVE_EVIDENCE_MIN_TOP_SCORE': '0.2',
        'ADAPTIVE_EVIDENCE_MIN_MEAN_SCORE': '0.1',
        'ADAPTIVE_EVIDENCE_FAIL_OPEN_WHEN_UNSCORED': 'false',
    })
    assert cfg == EvidenceConfig(top_k=2, min_top_score=0.2, min_mean_score=0.1,
                                 fail_open_when_unscored=False)


def test_config_from_empty_env_uses_documented_defaults():
    cfg = EvidenceConfig.from_env({})
    assert cfg.min_top_score == DEFAULT_MIN_TOP_SCORE
    assert cfg.min_mean_score is None
    assert cfg.fail_open_when_unscored is True


@pytest.mark.parametrize('env', [
    {'ADAPTIVE_EVIDENCE_MIN_TOP_SCORE': 'high'},
    {'ADAPTIVE_EVIDENCE_MIN_TOP_SCORE': 'nan'},
    {'ADAPTIVE_EVIDENCE_TOP_K': '2.5'},
    {'ADAPTIVE_EVIDENCE_FAIL_OPEN_WHEN_UNSCORED': 'maybe'},
])
def test_malformed_env_fails_loudly_instead_of_silently_defaulting(env):
    with pytest.raises(ValueError, match='ADAPTIVE_EVIDENCE'):
        EvidenceConfig.from_env(env)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        EvidenceConfig(top_k=0)
    with pytest.raises(ValueError):
        EvidenceConfig(min_top_score=float('inf'))


# ── requirement 7: scores are ranking signals, not probabilities ─────────────
def test_scores_outside_unit_interval_are_compared_raw_not_clamped_or_assumed_probabilities():
    # A probability-based implementation would reject/clamp these. A raw
    # ranking-score comparison handles any scale.
    cfg = EvidenceConfig(min_top_score=5.0)
    assert EvidenceEvaluator(cfg).evaluate(chunks_with(12.5, 3.0)).sufficient
    assert not EvidenceEvaluator(cfg).evaluate(chunks_with(4.9, 3.0)).sufficient
    assert EvidenceEvaluator(EvidenceConfig(min_top_score=-2.0)).evaluate(
        chunks_with(-1.5, -6.0)).sufficient


def test_verdict_never_calls_a_score_a_probability_or_confidence():
    v = EvidenceEvaluator().evaluate(strong_chunks())
    names = {f.name for f in dataclasses.fields(v)} | set(v.to_dict())
    assert not any('prob' in n or 'confid' in n or 'likel' in n for n in names), names


def test_module_does_not_assume_a_unit_interval():
    src = inspect.getsource(evidence.EvidenceEvaluator)
    assert 'clamp' not in src.lower() and 'min(1' not in src and 'max(0' not in src


def test_default_is_not_presented_as_a_probability_threshold():
    # The placeholder must stay far from the "0.7 probability" trap and be
    # documented as uncalibrated.
    assert DEFAULT_MIN_TOP_SCORE != 0.7
    assert 'UNCALIBRATED' in inspect.getsource(evidence)


# ── synthetic / missing scores are never judged ──────────────────────────────
def test_unreranked_fallback_scores_are_not_evaluated_and_pass_by_default():
    # The fallback reranker emits 1.0, 0.9, ... purely from rank order.
    chunks = [make_chunk(f'c{i}', 1.0 - i * 0.1, reranked=False) for i in range(3)]
    v = EvidenceEvaluator(EvidenceConfig(min_top_score=0.99)).evaluate(chunks)
    assert v.evaluated is False
    assert v.sufficient is True
    assert v.reason == REASON_UNSCORED_ACCEPTED
    assert v.top_score is None            # synthetic scores are not reported as evidence
    assert not v.retryable


def test_unreranked_scores_can_be_configured_to_fail_closed():
    chunks = [make_chunk('c0', 1.0, reranked=False)]
    v = EvidenceEvaluator(EvidenceConfig(fail_open_when_unscored=False)).evaluate(chunks)
    assert v.evaluated is False and v.sufficient is False
    assert v.reason == REASON_UNSCORED_REFUSED
    assert not v.retryable                 # retrying cannot produce real scores


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), 'high', True])
def test_missing_or_non_numeric_scores_are_treated_as_unscored(bad):
    v = EvidenceEvaluator(EvidenceConfig(min_top_score=0.99)).evaluate(
        [make_chunk('c0', 0.9), make_chunk('c1', bad)])
    assert v.evaluated is False
