"""D5-Lite: the `performance` block schema (only measured fields) and the log line."""
import json
import logging
from types import SimpleNamespace

from performance.pricing import PricingConfig
from performance.recorder import RequestRecorder
from performance.report import (
    LOGGER_NAME, build_performance, log_performance, routing_path_from_adaptive,
)
from performance.tokens import TokenUsage


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_total_only_when_nothing_else_was_measured():
    perf = build_performance(RequestRecorder())
    assert set(perf) == {'total_latency_ms'}                  # no stages / tokens / cost / path fabricated


def test_full_block_has_the_documented_shape_and_order():
    clock = Clock()
    rec = RequestRecorder(clock=clock)
    for name, secs in [('generation', 0.4), ('dense_retrieval', 0.1), ('routing', 0.001)]:
        with rec.stage(name):
            clock.now += secs
    rec.record_llm_call('generation', TokenUsage(1000, 500, 1500))
    perf = build_performance(rec, routing_path='direct',
                             pricing=PricingConfig(input_per_1m=0.15, output_per_1m=0.60))
    assert list(perf) == ['total_latency_ms', 'stages', 'tokens', 'estimated_cost', 'cost_currency', 'routing_path']
    assert list(perf['stages']) == ['dense_retrieval_ms', 'routing_ms', 'generation_ms']   # pipeline order
    assert perf['stages']['generation_ms'] == 400.0
    assert perf['tokens'] == {'input': 1000, 'output': 500, 'total': 1500}
    assert perf['estimated_cost'] == 0.00045 and perf['cost_currency'] == 'USD'
    assert perf['routing_path'] == 'direct'
    json.dumps(perf)                                           # serializable as-is


def test_tokens_without_pricing_give_no_cost():
    rec = RequestRecorder()
    rec.record_llm_call('generation', TokenUsage(10, 5, 15))
    perf = build_performance(rec, pricing=PricingConfig())
    assert perf['tokens']['total'] == 15
    assert 'estimated_cost' not in perf and 'cost_currency' not in perf


def test_call_without_reported_usage_gives_no_tokens_and_no_cost():
    rec = RequestRecorder()
    rec.record_llm_call('generation', None)
    perf = build_performance(rec, pricing=PricingConfig(input_per_1m=1, output_per_1m=1))
    assert 'tokens' not in perf and 'estimated_cost' not in perf


def test_zero_reported_tokens_yield_zero_cost_not_a_missing_field():
    rec = RequestRecorder()
    rec.record_llm_call('generation', TokenUsage(0, 0, 0))
    perf = build_performance(rec, pricing=PricingConfig(input_per_1m=1, output_per_1m=1))
    assert perf['tokens'] == {'input': 0, 'output': 0, 'total': 0} and perf['estimated_cost'] == 0.0


def test_routing_path_is_the_final_attempt_kind():
    attempts = lambda *kinds: SimpleNamespace(attempts=[SimpleNamespace(kind=k) for k in kinds])
    assert routing_path_from_adaptive(attempts('direct')) == 'direct'
    assert routing_path_from_adaptive(attempts('decomposed')) == 'decomposed'
    assert routing_path_from_adaptive(attempts('direct', 'reformulated')) == 'reformulated'
    assert routing_path_from_adaptive(attempts('decomposed', 'reformulated')) == 'reformulated'
    assert routing_path_from_adaptive(SimpleNamespace()) is None


def test_log_line_is_structured_and_contains_only_measurements(caplog):
    perf = {'total_latency_ms': 12.5, 'stages': {'generation_ms': 10.0}, 'routing_path': 'direct'}
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_performance(perf, refused=False)
    (record,) = caplog.records
    line = json.loads(record.getMessage())
    assert line == {'event': 'rag_query_performance', 'refused': False, **perf}


def test_logging_is_silent_unless_info_is_enabled(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        log_performance({'total_latency_ms': 1.0})
    assert caplog.records == []
