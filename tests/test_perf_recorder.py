"""D5-Lite: latency instrumentation (stage timers, request scoping)."""
import asyncio
from types import SimpleNamespace

import pytest

from performance.recorder import (
    RERANKING, RERANK_THROTTLE_WAIT, RequestRecorder, activate, current_recorder,
    deactivate, record_llm_usage, stage,
)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


def test_stage_measures_elapsed_milliseconds(clock):
    rec = RequestRecorder(clock=clock)
    with rec.stage('dense_retrieval'):
        clock.advance(0.25)
    assert rec.stages_ms == {'dense_retrieval': 250.0}


def test_repeated_stage_is_summed_and_counted(clock):
    rec = RequestRecorder(clock=clock)
    for seconds in (0.1, 0.2):
        with rec.stage(RERANKING):
            clock.advance(seconds)
    assert rec.stages_ms[RERANKING] == pytest.approx(300.0)
    assert rec.stage_calls(RERANKING) == 2


def test_nested_stages_are_independent_measurements(clock):
    rec = RequestRecorder(clock=clock)
    with rec.stage(RERANKING):
        clock.advance(0.1)
        with rec.stage(RERANK_THROTTLE_WAIT):
            clock.advance(0.5)
    assert rec.stages_ms[RERANK_THROTTLE_WAIT] == pytest.approx(500.0)
    assert rec.stages_ms[RERANKING] == pytest.approx(600.0)   # the wait is INSIDE reranking


def test_stage_is_recorded_even_when_the_block_raises(clock):
    rec = RequestRecorder(clock=clock)
    with pytest.raises(RuntimeError):
        with rec.stage('generation'):
            clock.advance(0.05)
            raise RuntimeError('boom')
    assert rec.stages_ms['generation'] == pytest.approx(50.0)


def test_total_elapsed_runs_from_creation(clock):
    rec = RequestRecorder(clock=clock)
    clock.advance(1.5)
    assert rec.elapsed_ms() == pytest.approx(1500.0)


def test_unmeasured_stage_is_absent_not_zero():
    assert 'routing' not in RequestRecorder().stages_ms


def test_module_stage_is_a_noop_without_an_active_recorder():
    assert current_recorder() is None
    with stage('dense_retrieval'):
        pass                                   # must not raise or record anywhere
    record_llm_usage('generation', SimpleNamespace(usage=None))   # also a no-op


def test_activate_and_deactivate_scope_the_recorder(clock):
    outer, inner = RequestRecorder(clock=clock), RequestRecorder(clock=clock)
    t1 = activate(outer)
    t2 = activate(inner)
    with stage('fusion'):
        clock.advance(0.01)
    assert 'fusion' in inner.stages_ms and 'fusion' not in outer.stages_ms
    deactivate(t2)
    assert current_recorder() is outer
    deactivate(t1)
    assert current_recorder() is None


def test_concurrent_requests_do_not_share_a_recorder():
    async def one_request(name, results):
        rec = RequestRecorder()
        token = activate(rec)
        try:
            await asyncio.sleep(0)             # let the other task run in between
            with stage(name):
                await asyncio.sleep(0)
            assert current_recorder() is rec
            results[name] = set(rec.stages_ms)
        finally:
            deactivate(token)

    async def main():
        results = {}
        await asyncio.gather(one_request('dense_retrieval', results), one_request('bm25_retrieval', results))
        return results

    assert asyncio.run(main()) == {'dense_retrieval': {'dense_retrieval'}, 'bm25_retrieval': {'bm25_retrieval'}}


def test_record_llm_usage_never_raises_on_a_malformed_response():
    class Hostile:
        @property
        def usage(self):
            raise RuntimeError('no usage for you')

    rec = RequestRecorder()
    token = activate(rec)
    try:
        record_llm_usage('generation', Hostile())
    finally:
        deactivate(token)
    assert rec.llm_calls == [('generation', None)]       # the call is noted, its usage is unknown
