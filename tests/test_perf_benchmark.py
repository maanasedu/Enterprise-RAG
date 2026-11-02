"""D5-Lite: benchmark statistics, selection, concurrency, failed requests."""
import json
import threading
import time

import httpx
import numpy as np
import pytest

import scripts.benchmark_performance as bench
from evaluation.retrieval_eval import BenchmarkQuestion, HeldOutSplitError
from performance.stats import percentile, sample_size_warnings, summarize


def Q(i, category='factual'):
    return BenchmarkQuestion(id=f'q{i}', category=category, question=f'secret question text {i}?',
                             answerable=True, evidence_status='sufficient',
                             gold_chunk_ids=('c',), gold_document_ids=('d',))


def result(i, ok=True, latency=100.0, perf=None, refused=False, error=None):
    return bench.RequestResult(index=i, id=f'q{i}', category='factual', ok=ok, client_latency_ms=latency,
                               status_code=200 if ok else 500, error=error, refused=refused if ok else None,
                               performance=perf)


# ── statistics ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize('values', [[5], [1, 2], [3, 1, 2], list(range(1, 101)), [10, 10, 10, 200], [0.5, 2.5, 9.0, 4.0, 7.5]])
@pytest.mark.parametrize('q', [0, 50, 95, 99, 100])
def test_percentile_matches_numpy_linear_interpolation(values, q):
    assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))


def test_percentile_edge_cases():
    assert percentile([], 50) is None
    with pytest.raises(ValueError):
        percentile([1], 101)
    with pytest.raises(ValueError):
        percentile([1, float('nan')], 50)


def test_summarize_known_values_and_empty_sample():
    s = summarize([100, 200, 300, 400])
    assert (s['n'], s['mean'], s['min'], s['max'], s['p50']) == (4, 250.0, 100.0, 400.0, 250.0)
    assert s['p95'] == pytest.approx(385.0) and s['p99'] == pytest.approx(397.0)
    empty = summarize([])
    assert empty['n'] == 0 and all(empty[k] is None for k in ('mean', 'p50', 'p95', 'p99', 'min', 'max'))


def test_small_samples_get_percentile_warnings():
    assert any('P95' in w for w in sample_size_warnings(5))
    assert sample_size_warnings(0) == ['no successful requests: no latency statistics']
    assert sample_size_warnings(500) == []


# ── question selection reuses existing benchmark data, deterministically ─────
def test_selection_is_deterministic_filtered_and_in_file_order():
    qs = [Q(0, 'factual'), Q(1, 'adversarial'), Q(2, 'numerical'), Q(3, 'factual')]
    assert [q.id for q in bench.select_questions(qs, limit=2)] == ['q0', 'q1']
    assert [q.id for q in bench.select_questions(qs, exclude_categories=['adversarial'])] == ['q0', 'q2', 'q3']
    assert [q.id for q in bench.select_questions(qs, categories=['factual'], limit=5)] == ['q0', 'q3']
    assert bench.select_questions(qs, limit=0) == []


def test_real_benchmark_files_load_through_the_dimension_1_loader():
    from evaluation.retrieval_eval import load_questions, split_path
    questions = load_questions(split_path('data/evaluation', 'calibration'))
    picked = bench.select_questions(questions, limit=7, exclude_categories=['adversarial'])
    assert len(picked) == 7 and all(q.question for q in picked)           # no hand-written questions needed
    assert [q.id for q in picked] == [q.id for q in bench.select_questions(questions, limit=7,
                                                                            exclude_categories=['adversarial'])]


def test_held_out_test_split_still_needs_explicit_confirmation(capsys):
    with pytest.raises(HeldOutSplitError):
        from evaluation.retrieval_eval import guard_split
        guard_split('test', False)
    assert bench.main(['--split', 'test']) == 2                            # CLI refuses without the flag
    assert 'held out' in capsys.readouterr().err


# ── running ──────────────────────────────────────────────────────────────────
def test_every_question_is_sent_once_in_order_at_any_concurrency():
    seen = []
    lock = threading.Lock()

    def send(i, q):
        with lock:
            seen.append(q.id)
        time.sleep(0.005 * (5 - i % 5))                 # finish out of order on purpose
        return result(i)

    run = bench.run_benchmark([Q(i) for i in range(12)], send, concurrency=4)
    assert sorted(seen) == sorted(f'q{i}' for i in range(12))
    assert [r.id for r in run.results] == [f'q{i}' for i in range(12)]     # reported in question order


def test_concurrency_is_actually_parallel_and_bounded():
    active, peak, lock = 0, 0, threading.Lock()

    def send(i, q):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return result(i)

    bench.run_benchmark([Q(i) for i in range(8)], send, concurrency=3)
    assert 2 <= peak <= 3


def test_invalid_concurrency_and_warmup_are_rejected():
    with pytest.raises(ValueError):
        bench.run_benchmark([Q(0)], lambda i, q: result(i), concurrency=0)
    with pytest.raises(ValueError):
        bench.run_benchmark([Q(0)], lambda i, q: result(i), warmup=-1)


def test_warmup_requests_are_sent_but_not_counted():
    sent = []
    run = bench.run_benchmark([Q(i) for i in range(5)], lambda i, q: (sent.append(i), result(i))[1], warmup=2)
    assert len(sent) == 7 and len(run.results) == 5 and run.warmup_requests == 2


def test_a_sender_that_raises_is_recorded_as_a_failed_request_not_a_crash():
    def send(i, q):
        if i == 1:
            raise RuntimeError('sender bug')
        return result(i)

    run = bench.run_benchmark([Q(i) for i in range(3)], send)
    assert [r.ok for r in run.results] == [True, False, True] and run.results[1].error == 'RuntimeError'


def test_throughput_uses_the_injected_clock():
    ticks = iter([10.0, 14.0])                         # start, end -> 4.0 s of wall time
    run = bench.run_benchmark([Q(i) for i in range(8)], lambda i, q: result(i), clock=lambda: next(ticks))
    summary = bench.summarize_run(run)
    assert run.wall_seconds == 4.0 and summary['throughput_rps'] == 2.0      # 8 successful / 4 s


# ── summary statistics ───────────────────────────────────────────────────────
def perf(total, stages, path='direct'):
    return {'total_latency_ms': total, 'stages': stages, 'routing_path': path}


def test_summary_counts_latency_and_stage_breakdown():
    results = [
        result(0, latency=100.0, perf=perf(90.0, {'dense_retrieval_ms': 10.0, 'generation_ms': 60.0})),
        result(1, latency=300.0, perf=perf(280.0, {'dense_retrieval_ms': 30.0, 'generation_ms': 200.0, 'reformulation_ms': 5.0}, 'reformulated')),
        result(2, ok=False, latency=50.0, error='ReadTimeout'),
        result(3, latency=200.0, perf=perf(180.0, {'dense_retrieval_ms': 20.0})),
    ]
    s = bench.summarize_run(bench.BenchmarkRun(results, wall_seconds=2.0))
    assert (s['total_requests'], s['successful_requests'], s['failed_requests']) == (4, 3, 1)
    assert s['throughput_rps'] == 1.5
    assert s['client_latency_ms']['mean'] == 200.0 and s['client_latency_ms']['p50'] == 200.0   # failed request excluded
    assert s['server_total_latency_ms']['mean'] == 183.33
    stages = s['stage_latency_ms']
    assert stages['dense_retrieval_ms']['n'] == 3 and stages['dense_retrieval_ms']['mean'] == 20.0
    assert stages['generation_ms']['n'] == 2 and stages['reformulation_ms']['n'] == 1   # absent != zero
    assert s['routing_paths'] == {'direct': 2, 'reformulated': 1}
    assert s['errors'] == {'ReadTimeout': 1}


def test_all_requests_failed_gives_no_statistics_not_zeros():
    s = bench.summarize_run(bench.BenchmarkRun([result(i, ok=False, error='ConnectError') for i in range(3)], 1.0))
    assert s['successful_requests'] == 0 and s['throughput_rps'] == 0.0
    assert s['client_latency_ms']['p95'] is None and s['stage_latency_ms'] == {}


def test_empty_run_and_zero_wall_time_do_not_divide_by_zero():
    s = bench.summarize_run(bench.BenchmarkRun([], 0.0))
    assert s['total_requests'] == 0 and s['throughput_rps'] is None


def test_successful_responses_without_a_performance_block_are_counted_separately():
    s = bench.summarize_run(bench.BenchmarkRun([result(0, perf=None), result(1, perf=perf(10.0, {}))], 1.0))
    assert s['successful_requests'] == 2 and s['responses_without_performance_block'] == 1
    assert s['server_total_latency_ms']['n'] == 1


def test_refusals_are_successful_but_counted():
    s = bench.summarize_run(bench.BenchmarkRun([result(0, refused=True), result(1)], 1.0))
    assert s['successful_requests'] == 2 and s['refused_responses'] == 1


# ── the HTTP sender (failed requests, redaction) ─────────────────────────────
def sender_for(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return bench.make_http_sender('http://api.test', top_k=5, timeout_s=5, run_id='r1', client=client)


def test_http_sender_success_sends_the_documented_payload_and_reads_performance():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={'answer': 'A', 'refused': False,
                                         'performance': {'total_latency_ms': 12.0, 'routing_path': 'direct'}})

    r = sender_for(handler)(3, Q(3))
    assert r.ok and r.performance['routing_path'] == 'direct' and r.client_latency_ms >= 0
    assert captured == {'question': 'secret question text 3?', 'top_k': 5, 'include_performance': True,
                        'session_id': 'perf-r1-3'}                         # unique session per request


def test_http_sender_failures_are_recorded_without_leaking_response_bodies():
    r500 = sender_for(lambda req: httpx.Response(500, json={'detail': 'LEAK: internal stack trace'}))(0, Q(0))
    assert not r500.ok and r500.error == 'HTTP 500' and r500.status_code == 500
    assert 'LEAK' not in json.dumps(r500.to_dict())

    def timeout(request):
        raise httpx.ReadTimeout('too slow', request=request)

    assert sender_for(timeout)(0, Q(0)).error == 'ReadTimeout'
    assert sender_for(lambda req: httpx.Response(200, content=b'not json'))(0, Q(0)).error == 'invalid JSON response'
    assert sender_for(lambda req: httpx.Response(200, json=['not', 'an', 'object']))(0, Q(0)).error == 'invalid JSON response'


def test_report_records_never_contain_question_or_answer_text():
    def handler(request):
        return httpx.Response(200, json={'answer': 'ANSWER-TEXT', 'sources': [{'text': 'DOC-TEXT'}],
                                         'performance': {'total_latency_ms': 1.0}})

    run = bench.run_benchmark([Q(0)], sender_for(handler))
    dumped = json.dumps([r.to_dict() for r in run.results])
    assert 'secret question text' not in dumped and 'ANSWER-TEXT' not in dumped and 'DOC-TEXT' not in dumped


def test_credentials_are_stripped_from_the_stored_base_url():
    assert bench.redact_url('http://user:pass@localhost:8000/api?token=abc') == 'http://localhost:8000/api'


def test_cli_exits_2_when_the_server_is_unreachable(capsys, monkeypatch):
    def refuse(*a, **k):
        raise httpx.ConnectError('refused')

    monkeypatch.setattr(httpx, 'get', refuse)
    assert bench.main(['--limit', '1']) == 2
    assert 'cannot reach' in capsys.readouterr().err
