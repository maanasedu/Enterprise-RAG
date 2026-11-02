"""D5-Lite: token accounting, cost calculation, pricing configuration, and the
zero / missing-token edge cases."""
import json
from types import SimpleNamespace

import pytest

from performance.config import PerformanceConfig
from performance.pricing import PricingConfig
from performance.tokens import TokenUsage, aggregate_usage, extract_usage


def resp(**usage):
    return SimpleNamespace(usage=SimpleNamespace(**usage))


# ── reading provider-reported usage ──────────────────────────────────────────
def test_extracts_openai_groq_style_usage():
    u = extract_usage(resp(prompt_tokens=120, completion_tokens=30, total_tokens=150, queue_time=0.01))
    assert u == TokenUsage(input=120, output=30, total=150)


def test_extracts_anthropic_style_usage_and_dict_usage():
    assert extract_usage(resp(input_tokens=7, output_tokens=3)) == TokenUsage(7, 3, None)
    assert extract_usage({'usage': {'prompt_tokens': 5, 'completion_tokens': 2}}) == TokenUsage(5, 2, None)


@pytest.mark.parametrize('response', [
    SimpleNamespace(),                                   # no usage attribute
    SimpleNamespace(usage=None),
    resp(prompt_tokens=None, completion_tokens=None, total_tokens=None),
    resp(prompt_tokens=-5, completion_tokens='lots', total_tokens=True),   # garbage is not a count
    resp(prompt_tokens=float('nan'), completion_tokens=float('inf')),
])
def test_missing_or_garbage_usage_is_none_never_invented(response):
    assert extract_usage(response) is None


def test_zero_tokens_reported_by_the_provider_is_a_real_value():
    u = extract_usage(resp(prompt_tokens=0, completion_tokens=0, total_tokens=0))
    assert u == TokenUsage(0, 0, 0) and u.complete


def test_integer_valued_float_counts_are_accepted():
    assert extract_usage(resp(prompt_tokens=10.0, completion_tokens=4.0)) == TokenUsage(10, 4, None)


# ── aggregating over a request's LLM calls ───────────────────────────────────
def test_aggregate_single_generation_call():
    agg = aggregate_usage([('generation', TokenUsage(100, 20, 120))])
    assert agg == {'input': 100, 'output': 20, 'total': 120}


def test_aggregate_total_falls_back_to_input_plus_output():
    assert aggregate_usage([('generation', TokenUsage(100, 20, None))])['total'] == 120


def test_aggregate_sums_calls_and_breaks_down_by_purpose():
    agg = aggregate_usage([('reformulation', TokenUsage(40, 10, 50)), ('generation', TokenUsage(100, 20, 120))])
    assert (agg['input'], agg['output'], agg['total']) == (140, 30, 170)
    assert agg['by_purpose']['reformulation']['total'] == 50
    assert 'incomplete' not in agg


def test_aggregate_returns_none_when_no_call_reported_usage():
    assert aggregate_usage([]) is None
    assert aggregate_usage([('generation', None)]) is None
    assert aggregate_usage([('generation', TokenUsage(input=5))]) is None     # half-reported: unusable


def test_aggregate_flags_incomplete_when_a_call_lacks_usage():
    agg = aggregate_usage([('reformulation', None), ('generation', TokenUsage(10, 5, 15))])
    assert agg['total'] == 15 and agg['incomplete'] is True


# ── cost calculation ─────────────────────────────────────────────────────────
def test_cost_is_tokens_times_configured_price_per_million():
    p = PricingConfig(input_per_1m=0.15, output_per_1m=0.60)
    assert p.estimate_cost(1000, 500) == pytest.approx(0.00045)
    assert p.estimate_cost(1_000_000, 1_000_000) == pytest.approx(0.75)


def test_zero_tokens_cost_zero_but_missing_tokens_cost_nothing_known():
    p = PricingConfig(input_per_1m=1.0, output_per_1m=2.0)
    assert p.estimate_cost(0, 0) == 0.0
    assert p.estimate_cost(None, 10) is None
    assert p.estimate_cost(10, None) is None


def test_no_pricing_means_no_estimate():
    assert PricingConfig().estimate_cost(1000, 1000) is None
    assert not PricingConfig(input_per_1m=1.0).available      # half a price is not a price


# ── pricing configuration ────────────────────────────────────────────────────
def test_env_prices_configure_pricing():
    p = PricingConfig.from_env({'LLM_PRICE_INPUT_PER_1M': '0.2', 'LLM_PRICE_OUTPUT_PER_1M': '0.8',
                                'LLM_PRICE_CURRENCY': 'EUR'})
    assert (p.input_per_1m, p.output_per_1m, p.currency, p.source) == (0.2, 0.8, 'EUR', 'env')


def test_env_prices_must_be_set_together():
    with pytest.raises(ValueError, match='set together'):
        PricingConfig.from_env({'LLM_PRICE_INPUT_PER_1M': '0.2'})


@pytest.mark.parametrize('bad', ['abc', '-1', 'nan', 'inf'])
def test_malformed_or_negative_env_price_fails_loudly(bad):
    with pytest.raises(ValueError, match='LLM_PRICE'):
        PricingConfig.from_env({'LLM_PRICE_INPUT_PER_1M': bad, 'LLM_PRICE_OUTPUT_PER_1M': '1'})


def write_pricing(tmp_path, models, **top):
    path = tmp_path / 'pricing.json'
    path.write_text(json.dumps({'currency': 'USD', 'models': models, **top}))
    return str(path)


def test_pricing_file_entry_is_found_by_provider_and_model(tmp_path):
    f = write_pricing(tmp_path, {'groq/some-model': {'input_per_1m': 1, 'output_per_1m': 3, 'as_of': '2026-01-01'}})
    p = PricingConfig.from_env({'LLM_PRICING_FILE': f}, provider='groq', model='some-model')
    assert (p.input_per_1m, p.output_per_1m, p.as_of) == (1.0, 3.0, '2026-01-01')
    assert p.source.endswith(':groq/some-model')


def test_env_overrides_the_pricing_file(tmp_path):
    f = write_pricing(tmp_path, {'groq/m': {'input_per_1m': 1, 'output_per_1m': 3}})
    p = PricingConfig.from_env({'LLM_PRICING_FILE': f, 'LLM_PRICE_INPUT_PER_1M': '9', 'LLM_PRICE_OUTPUT_PER_1M': '9'},
                               provider='groq', model='m')
    assert p.input_per_1m == 9.0 and p.source == 'env'


def test_unknown_model_or_unknown_provider_is_unpriced_not_guessed(tmp_path):
    f = write_pricing(tmp_path, {'groq/m': {'input_per_1m': 1, 'output_per_1m': 3}})
    assert not PricingConfig.from_env({'LLM_PRICING_FILE': f}, provider='groq', model='other').available
    assert not PricingConfig.from_env({'LLM_PRICING_FILE': f}, provider=None, model='m').available


def test_bad_pricing_file_fails_loudly(tmp_path):
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json')
    with pytest.raises(ValueError, match='not valid JSON'):
        PricingConfig.from_env({'LLM_PRICING_FILE': str(bad)}, provider='groq', model='m')
    with pytest.raises(ValueError, match='file not found'):
        PricingConfig.from_env({'LLM_PRICING_FILE': str(tmp_path / 'missing.json')})
    neg = write_pricing(tmp_path, {'groq/m': {'input_per_1m': -1, 'output_per_1m': 3}})
    with pytest.raises(ValueError, match='non-negative'):
        PricingConfig.from_env({'LLM_PRICING_FILE': neg}, provider='groq', model='m')


def test_shipped_pricing_file_parses_and_is_flagged_unverified():
    p = PricingConfig.from_env({}, provider='groq', model='openai/gpt-oss-120b')
    assert p.available and p.as_of and 'UNVERIFIED' in (p.note or '')
    # pricing lives in data, not code: the model is unpriced under any other name
    assert not PricingConfig.from_env({}, provider='groq', model='fake-model').available


def test_performance_config_defaults_to_opt_in_and_parses_the_flag():
    assert PerformanceConfig.from_env({}).include_in_response is False
    assert PerformanceConfig.from_env({'PERFORMANCE_METADATA_ENABLED': 'true'}).include_in_response is True
    with pytest.raises(ValueError, match='PERFORMANCE_METADATA_ENABLED'):
        PerformanceConfig.from_env({'PERFORMANCE_METADATA_ENABLED': 'maybe'})
