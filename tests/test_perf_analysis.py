"""D5-Lite: cost/performance analysis, per routing path and baseline vs adaptive."""
import json

import pytest

import scripts.analyze_performance as an


def rec(i, path='direct', total=100.0, tokens=(100, 20), cost=None, refused=False, stages=None, ok=True,
        throttle=None, currency='USD', generated=True):
    perf = {'total_latency_ms': total, 'stages': dict(stages or {})}
    if generated:
        perf['stages'].setdefault('generation_ms', 10.0)
    if throttle is not None:
        perf['stages']['rerank_throttle_wait_ms'] = throttle
    if path:
        perf['routing_path'] = path
    if tokens:
        perf['tokens'] = {'input': tokens[0], 'output': tokens[1], 'total': tokens[0] + tokens[1]}
    if cost is not None:
        perf['estimated_cost'], perf['cost_currency'] = cost, currency
    return {'id': f'q{i}', 'ok': ok, 'refused': refused, 'performance': perf if ok else None}


def refusal(i, path='reformulated', total=400.0):
    r = rec(i, path=path, total=total, tokens=None, refused=True, generated=False)
    return r


def run(records, label='x', **protocol):
    return {'label': label, 'protocol': {'split': 'calibration', 'n_questions': len(records), 'limit': 20,
                                         'categories': None, 'exclude_categories': None, 'top_k': 5, **protocol},
            'per_request': records}


# ── by routing path ──────────────────────────────────────────────────────────
def test_by_path_counts_shares_latency_tokens_and_cost():
    records = [rec(0, 'direct', 100.0, (100, 20), 0.001), rec(1, 'direct', 200.0, (300, 40), 0.003),
               rec(2, 'decomposed', 500.0, (900, 60), 0.009), rec(3, 'decomposed', 700.0, (1100, 80), 0.011)]
    a = an.analyze_run(run(records))
    direct, decomposed = a['by_routing_path']['direct'], a['by_routing_path']['decomposed']
    assert (direct['n'], direct['share_pct']) == (2, 50.0)
    assert direct['latency_ms']['mean'] == 150.0 and direct['latency_ms']['p50'] == 150.0
    assert direct['tokens'] == {'n_reported': 2, 'avg_input': 200.0, 'avg_output': 30.0, 'avg_total': 230.0, 'n_incomplete': 0}
    assert direct['cost']['total'] == pytest.approx(0.004) and direct['cost']['avg_per_priced_query'] == pytest.approx(0.002)
    assert decomposed['cost']['total'] == pytest.approx(0.02)
    assert a['overall']['cost']['total'] == pytest.approx(0.024)           # sum of the path costs


def test_refusals_make_no_llm_call_and_do_not_distort_cost_per_priced_query():
    records = [rec(0, 'direct', 100.0, (100, 20), 0.002), refusal(1), refusal(2)]
    s = an.analyze_run(run(records))['overall']
    assert s['refused'] == 2 and s['cost']['n_no_llm_call'] == 2 and s['cost']['n_unpriced'] == 0
    assert s['cost']['avg_per_priced_query'] == pytest.approx(0.002)       # mean over queries that were priced
    assert s['cost']['avg_per_query_all'] == pytest.approx(0.002 / 3, abs=1e-8)   # refusals count as zero LLM cost


def test_unpriced_generated_queries_block_the_all_queries_average():
    records = [rec(0, tokens=(100, 20), cost=0.002), rec(1, tokens=None, cost=None)]   # generated, usage missing
    c = an.analyze_run(run(records))['overall']['cost']
    assert c['n_unpriced'] == 1 and c['avg_per_query_all'] is None        # would silently understate, so withheld
    assert c['avg_per_priced_query'] == pytest.approx(0.002)


def test_missing_tokens_and_costs_are_none_not_zero():
    s = an.analyze_run(run([rec(0, tokens=None, cost=None)]))['overall']
    assert s['tokens']['avg_total'] is None and s['cost']['total'] is None and s['cost']['avg_per_priced_query'] is None


def test_failed_and_performance_less_requests_are_excluded_and_counted():
    a = an.analyze_run(run([rec(0, cost=0.001), rec(1, ok=False), {'id': 'q2', 'ok': True, 'performance': None}]))
    assert a['usable_records'] == 1 and a['excluded_failed_or_without_performance'] == 2


def test_input_blocked_queries_form_their_own_no_retrieval_bucket():
    blocked = {'id': 'q9', 'ok': True, 'refused': True, 'performance': {'total_latency_ms': 3.0}}
    a = an.analyze_run(run([rec(0, cost=0.001), blocked]))
    assert set(a['by_routing_path']) == {'direct', 'no_retrieval'}
    assert a['by_routing_path']['no_retrieval']['cost']['n_no_llm_call'] == 1


def test_latency_excluding_the_rerank_pacing_sleep():
    a = an.analyze_run(run([rec(0, total=7000.0, throttle=6500.0, cost=0.001)]))['overall']
    assert a['latency_ms']['mean'] == 7000.0 and a['latency_excl_throttle_ms']['mean'] == 500.0


def test_report_text_is_neutral_and_flags_estimates():
    text = an.format_by_path(an.analyze_run(run([rec(i, cost=0.001) for i in range(3)])))
    assert 'estimates' in text and 'not billing data' in text and 'anecdotal' in text


# ── baseline vs adaptive ─────────────────────────────────────────────────────
def baseline_and_adaptive(adaptive_cost, adaptive_total=80.0):
    base = run([rec(i, 'full_retrieval', 100.0, (200, 20), 0.004) for i in range(4)], label='baseline')
    cand = run([rec(i, 'direct', adaptive_total, (200, 20), adaptive_cost) for i in range(4)], label='adaptive')
    return base, cand


def test_comparison_reports_measured_deltas_in_both_directions():
    base, cand = baseline_and_adaptive(adaptive_cost=0.005)                 # adaptive is MORE expensive
    c = an.compare_runs(base, cand)
    d = c['deltas']
    assert d['avg_cost_per_query_all']['delta'] == pytest.approx(0.001) and d['avg_cost_per_query_all']['delta_pct'] == 25.0
    assert d['mean_latency_ms']['delta'] == -20.0 and d['mean_latency_ms']['delta_pct'] == -20.0
    assert c['paired_answered_in_both']['n_paired'] == 4


def test_comparison_never_claims_savings_even_when_numbers_are_lower():
    base, cand = baseline_and_adaptive(adaptive_cost=0.001)
    c = an.compare_runs(base, cand)
    text = an.format_comparison(c).lower()
    assert c['deltas']['avg_cost_per_query_all']['delta'] < 0
    for forbidden in ('saves', 'saving', 'cheaper', 'improv', 'faster', 'reduces cost'):
        assert forbidden not in text
    assert 'descriptive only' in text and 'no significance test' in text


def test_refusal_differences_are_flagged_and_paired_comparison_excludes_refused_queries():
    base = run([rec(i, 'full_retrieval', 100.0, (200, 20), 0.004) for i in range(4)])
    cand = run([rec(0, 'direct', 90.0, (200, 20), 0.004), rec(1, 'direct', 90.0, (200, 20), 0.004),
                refusal(2), refusal(3)])
    c = an.compare_runs(base, cand)
    assert any('refusals differ' in w for w in c['warnings'])
    assert c['paired_answered_in_both']['n_paired'] == 2                    # only questions answered in both runs
    assert c['paired_answered_in_both']['mean_delta_cost'] == 0.0
    assert c['deltas']['avg_cost_per_query_all']['delta'] < 0               # overall differs only because of refusals


def test_mislabeled_or_mismatched_runs_are_warned_about():
    base, cand = baseline_and_adaptive(0.004)
    swapped = an.compare_runs(cand, base)
    assert any('ADAPTIVE_ROUTING_ENABLED=false' in w for w in swapped['warnings'])
    assert any('full_retrieval' in w for w in swapped['warnings'])
    other = run(base['per_request'], split='test')
    assert any("protocol differs on 'split'" in w for w in an.compare_runs(other, cand)['warnings'])


def test_comparison_with_an_unusable_run_does_not_crash():
    c = an.compare_runs(run([]), run([rec(0, cost=0.001)]))
    assert 'no usable records' in c['warnings'][0]
    assert 'unavailable' in an.format_comparison(c)


def test_cli_end_to_end_writes_json_and_returns_status(tmp_path, capsys):
    base, cand = baseline_and_adaptive(0.004)
    bp, cp, out = tmp_path / 'b.json', tmp_path / 'c.json', tmp_path / 'out.json'
    bp.write_text(json.dumps(base))
    cp.write_text(json.dumps(cand))
    assert an.main([str(cp), '--baseline', str(bp), '--output', str(out)]) == 0
    report = json.loads(out.read_text())
    assert report['analysis']['usable_records'] == 4 and 'comparison' in report
    assert an.main([str(tmp_path / 'missing.json')]) == 2
    bad = tmp_path / 'bad.json'
    bad.write_text('{"nope": 1}')
    assert an.main([str(bad)]) == 2
