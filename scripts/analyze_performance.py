"""D5-Lite -- cost / performance analysis of benchmark runs.

Reads the report written by scripts/benchmark_performance.py and summarizes,
per routing path (direct / decomposed / reformulated / full_retrieval /
no_retrieval): share of queries, latency (mean / P50 / P95), average
input / output / total tokens, and estimated cost per query and in total.

    python scripts/analyze_performance.py data/performance/benchmark_adaptive.json

Baseline comparison (normal/full retrieval vs adaptive routing). Run the
benchmark twice on the SAME split/limit/categories -- once against a server
started with ADAPTIVE_ROUTING_ENABLED=false (every query takes the legacy
`full_retrieval` path) and once with it enabled -- then:

    python scripts/analyze_performance.py data/performance/benchmark_adaptive.json \\
        --baseline data/performance/benchmark_baseline.json

This script reports measured differences only. It never says adaptive routing
"saves" anything: whether a difference exists, and why, is for the reader to
judge from the numbers, the paired subset and the warnings it prints.

Costs are the server's ESTIMATES (reported tokens x configured price), not
billing data. "Latency" here is the server-side `performance.total_latency_ms`
(excludes client/network queueing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance.stats import mean, summarize

NO_RETRIEVAL = 'no_retrieval'
FULL_RETRIEVAL = 'full_retrieval'
PROTOCOL_KEYS = ('split', 'n_questions', 'limit', 'categories', 'exclude_categories', 'top_k')
SMALL_SAMPLE = 30


# ── loading ──────────────────────────────────────────────────────────────────
def load_run(path: str) -> Dict:
    with open(path, encoding='utf-8') as f:
        run = json.load(f)
    if not isinstance(run, dict) or not isinstance(run.get('per_request'), list):
        raise ValueError(f'{path}: not a benchmark_performance report (no per_request list)')
    return run


def usable_records(run: Dict) -> List[Dict]:
    """Successful requests that carry a server `performance` block."""
    return [r for r in run['per_request']
            if r.get('ok') and isinstance(r.get('performance'), dict)
            and isinstance(r['performance'].get('total_latency_ms'), (int, float))]


def path_of(record: Dict) -> str:
    return record['performance'].get('routing_path') or NO_RETRIEVAL


def _stage(record: Dict, name: str) -> Optional[float]:
    value = (record['performance'].get('stages') or {}).get(name)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def latency_ms(record: Dict) -> float:
    return float(record['performance']['total_latency_ms'])


def latency_excl_throttle_ms(record: Dict) -> float:
    """Total minus the Cohere client-side rate-limit sleep (a fixed pacing
    delay, not retrieval work)."""
    return latency_ms(record) - (_stage(record, 'rerank_throttle_wait_ms') or 0.0)


def _tokens(record: Dict) -> Optional[Dict]:
    t = record['performance'].get('tokens')
    return t if isinstance(t, dict) else None


def _cost(record: Dict) -> Optional[float]:
    c = record['performance'].get('estimated_cost')
    return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else None


def _made_llm_call(record: Dict) -> bool:
    return _tokens(record) is not None or _stage(record, 'generation_ms') is not None


# ── summarizing ──────────────────────────────────────────────────────────────
def summarize_group(records: Sequence[Dict]) -> Dict:
    n = len(records)
    token_rows = [t for t in (_tokens(r) for r in records) if t is not None]
    costs = [c for c in (_cost(r) for r in records) if c is not None]
    n_no_llm = sum(1 for r in records if not _made_llm_call(r))
    n_unpriced = sum(1 for r in records if _made_llm_call(r) and _cost(r) is None)
    currencies = sorted({r['performance'].get('cost_currency') for r in records
                         if r['performance'].get('cost_currency')})
    total_cost = sum(costs) if costs else None

    def avg(key: str) -> Optional[float]:
        values = [t[key] for t in token_rows if isinstance(t.get(key), (int, float))]
        return round(mean(values), 2) if values else None

    return {
        'n': n,
        'refused': sum(1 for r in records if r.get('refused')),
        'latency_ms': summarize([latency_ms(r) for r in records]),
        'latency_excl_throttle_ms': summarize([latency_excl_throttle_ms(r) for r in records]),
        'tokens': {'n_reported': len(token_rows), 'avg_input': avg('input'),
                   'avg_output': avg('output'), 'avg_total': avg('total'),
                   'n_incomplete': sum(1 for t in token_rows if t.get('incomplete'))},
        'cost': {
            'currency': currencies[0] if len(currencies) == 1 else (currencies or None),
            'n_priced': len(costs),
            'n_no_llm_call': n_no_llm,
            'n_unpriced': n_unpriced,
            'total': round(total_cost, 8) if total_cost is not None else None,
            'avg_per_priced_query': round(total_cost / len(costs), 8) if costs else None,
            # Over ALL queries (refusals/blocked count as zero LLM cost). Only given
            # when no query made an LLM call we could not price; otherwise it would
            # silently understate.
            'avg_per_query_all': (round(total_cost / n, 8)
                                  if costs and n_unpriced == 0 and n else None),
        },
    }


def summarize_by_path(records: Sequence[Dict]) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = {}
    for r in records:
        groups.setdefault(path_of(r), []).append(r)
    total = len(records)
    out = {}
    for path in sorted(groups):
        s = summarize_group(groups[path])
        s['share_pct'] = round(100.0 * len(groups[path]) / total, 1) if total else None
        out[path] = s
    return out


def analyze_run(run: Dict) -> Dict:
    records = usable_records(run)
    return {
        'label': run.get('label'),
        'protocol': run.get('protocol', {}),
        'requests_in_report': len(run['per_request']),
        'usable_records': len(records),
        'excluded_failed_or_without_performance': len(run['per_request']) - len(records),
        'overall': summarize_group(records) if records else None,
        'by_routing_path': summarize_by_path(records),
    }


# ── baseline comparison ──────────────────────────────────────────────────────
def _delta(candidate: Optional[float], baseline: Optional[float]) -> Optional[Dict]:
    if candidate is None or baseline is None:
        return None
    diff = candidate - baseline
    pct = (diff / baseline * 100.0) if baseline not in (0, 0.0) else None
    return {'baseline': baseline, 'candidate': candidate,
            'delta': round(diff, 8), 'delta_pct': round(pct, 1) if pct is not None else None}


def paired_comparison(baseline_records: Sequence[Dict], candidate_records: Sequence[Dict]) -> Dict:
    """Same question in both runs, answered (not refused) in both: isolates the
    path effect from the refusal effect."""
    base = {r['id']: r for r in baseline_records if not r.get('refused')}
    cand = {r['id']: r for r in candidate_records if not r.get('refused')}
    ids = sorted(set(base) & set(cand))

    def mean_diff(getter) -> Optional[float]:
        diffs = []
        for i in ids:
            b, c = getter(base[i]), getter(cand[i])
            if b is not None and c is not None:
                diffs.append(c - b)
        return round(sum(diffs) / len(diffs), 8) if diffs else None

    return {
        'n_paired': len(ids),
        'mean_delta_latency_ms': mean_diff(latency_ms),
        'mean_delta_latency_excl_throttle_ms': mean_diff(latency_excl_throttle_ms),
        'mean_delta_total_tokens': mean_diff(lambda r: (_tokens(r) or {}).get('total')),
        'mean_delta_cost': mean_diff(_cost),
    }


def compare_runs(baseline: Dict, candidate: Dict) -> Dict:
    b_rec, c_rec = usable_records(baseline), usable_records(candidate)
    b_all, c_all = summarize_group(b_rec), summarize_group(c_rec)
    warnings: List[str] = []

    if not b_rec or not c_rec:
        return {'warnings': ['one of the runs has no usable records'], 'overall': None}

    for key in PROTOCOL_KEYS:
        if baseline.get('protocol', {}).get(key) != candidate.get('protocol', {}).get(key):
            warnings.append(f'protocol differs on {key!r}: baseline={baseline.get("protocol", {}).get(key)!r} '
                            f'vs candidate={candidate.get("protocol", {}).get(key)!r}; the runs may not be comparable')
    base_paths = {path_of(r) for r in b_rec}
    cand_paths = {path_of(r) for r in c_rec}
    if base_paths - {FULL_RETRIEVAL, NO_RETRIEVAL}:
        warnings.append(f'baseline contains adaptive paths {sorted(base_paths - {FULL_RETRIEVAL, NO_RETRIEVAL})}; '
                        f'was it run with ADAPTIVE_ROUTING_ENABLED=false?')
    if FULL_RETRIEVAL in cand_paths:
        warnings.append('candidate contains full_retrieval; was it run with ADAPTIVE_ROUTING_ENABLED=true?')
    if b_all['refused'] != c_all['refused']:
        warnings.append(
            f"refusals differ (baseline {b_all['refused']}, candidate {c_all['refused']}). A refused query makes no "
            f"LLM call, so overall latency/token/cost differences mix the routing effect with the refusal effect; "
            f"see the paired (answered-in-both) comparison.")
    if min(len(b_rec), len(c_rec)) < SMALL_SAMPLE:
        warnings.append(f'fewer than {SMALL_SAMPLE} usable queries in a run: treat differences as anecdotal')
    warnings.append('descriptive only: no significance test is applied; provider/API load, the Cohere pacing '
                    'sleep and run-to-run variation can all move these numbers')

    def pick(group: Dict, *keys):
        value = group
        for k in keys:
            value = value.get(k) if isinstance(value, dict) else None
        return value

    metrics = {
        'mean_latency_ms': ('latency_ms', 'mean'),
        'p50_latency_ms': ('latency_ms', 'p50'),
        'p95_latency_ms': ('latency_ms', 'p95'),
        'mean_latency_excl_throttle_ms': ('latency_excl_throttle_ms', 'mean'),
        'avg_total_tokens': ('tokens', 'avg_total'),
        'avg_cost_per_query_all': ('cost', 'avg_per_query_all'),
    }
    return {
        'baseline_overall': b_all, 'candidate_overall': c_all,
        'deltas': {name: _delta(pick(c_all, *keys), pick(b_all, *keys)) for name, keys in metrics.items()},
        'paired_answered_in_both': paired_comparison(b_rec, c_rec),
        'warnings': warnings,
    }


# ── formatting ───────────────────────────────────────────────────────────────
def _fmt(value, width: int = 10) -> str:
    return f"{'n/a' if value is None else value:>{width}}"


def format_by_path(analysis: Dict) -> str:
    lines = [f"Run: {analysis.get('label')}   usable={analysis['usable_records']} "
             f"excluded(failed / no performance block)={analysis['excluded_failed_or_without_performance']}"]
    if not analysis['by_routing_path']:
        return '\n'.join(lines + ['No usable records.'])
    head = (f"{'path':<14}{'n':>5}{'share%':>8}{'mean ms':>10}{'p50 ms':>10}{'p95 ms':>10}"
            f"{'in tok':>9}{'out tok':>9}{'tot tok':>9}{'$/priced':>12}{'cost total':>12}{'refused':>8}")
    lines += ['', head, '-' * len(head)]
    rows = list(analysis['by_routing_path'].items())
    if analysis['overall']:
        overall = dict(analysis['overall'], share_pct=100.0)
        rows.append(('ALL', overall))
    for path, s in rows:
        lat, tok, cost = s['latency_ms'], s['tokens'], s['cost']
        lines.append(f"{path:<14}{s['n']:>5}{_fmt(s['share_pct'], 8)}{_fmt(lat['mean'])}{_fmt(lat['p50'])}"
                     f"{_fmt(lat['p95'])}{_fmt(tok['avg_input'], 9)}{_fmt(tok['avg_output'], 9)}"
                     f"{_fmt(tok['avg_total'], 9)}{_fmt(cost['avg_per_priced_query'], 12)}"
                     f"{_fmt(cost['total'], 12)}{s['refused']:>8}")
    overall = analysis['overall']
    lines += ['', 'Latency = server-side performance.total_latency_ms. Costs are estimates (reported tokens x '
                 'configured price), not billing data.']
    if overall['cost']['n_no_llm_call']:
        lines.append(f"{overall['cost']['n_no_llm_call']} query(ies) made no LLM call (refused/blocked): "
                     f"no tokens, no LLM cost.")
    if overall['cost']['n_unpriced']:
        lines.append(f"{overall['cost']['n_unpriced']} query(ies) called the LLM but have no estimated cost "
                     f"(provider reported no usage, or pricing is not configured): costs above exclude them.")
    if overall['tokens']['n_incomplete']:
        lines.append(f"{overall['tokens']['n_incomplete']} query(ies) have incomplete token totals "
                     f"(a lower bound).")
    if overall['n'] < SMALL_SAMPLE:
        lines.append(f"n={overall['n']} < {SMALL_SAMPLE}: treat percentiles and path shares as anecdotal.")
    return '\n'.join(lines)


def format_comparison(comparison: Dict) -> str:
    if comparison.get('overall', 1) is None and 'deltas' not in comparison:
        return 'Comparison unavailable: ' + '; '.join(comparison['warnings'])
    names = {
        'mean_latency_ms': 'mean latency (ms)', 'p50_latency_ms': 'P50 latency (ms)',
        'p95_latency_ms': 'P95 latency (ms)',
        'mean_latency_excl_throttle_ms': 'mean latency excl. rerank pacing (ms)',
        'avg_total_tokens': 'avg total tokens/query',
        'avg_cost_per_query_all': 'est. LLM cost/query (all queries)',
    }
    lines = ['', 'Baseline (full retrieval) vs candidate (adaptive routing)', '',
             f"{'metric':<42}{'baseline':>14}{'candidate':>14}{'delta':>14}{'delta %':>9}"]
    lines.append('-' * len(lines[-1]))
    for key, label in names.items():
        d = comparison['deltas'][key]
        if d is None:
            lines.append(f"{label:<42}{'n/a':>14}{'n/a':>14}{'n/a':>14}{'n/a':>9}")
        else:
            lines.append(f"{label:<42}{d['baseline']:>14}{d['candidate']:>14}{d['delta']:>14}"
                         f"{_fmt(d['delta_pct'], 9)}")
    p = comparison['paired_answered_in_both']
    lines += ['', f"Paired, answered in both runs (n={p['n_paired']}), mean candidate-minus-baseline:",
              f"  latency {p['mean_delta_latency_ms']} ms | excl. pacing {p['mean_delta_latency_excl_throttle_ms']} ms | "
              f"total tokens {p['mean_delta_total_tokens']} | cost {p['mean_delta_cost']}"]
    lines += ['', 'Notes:'] + [f'  - {w}' for w in comparison['warnings']]
    return '\n'.join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description='D5-Lite cost/performance analysis of benchmark_performance reports.')
    p.add_argument('run', help='report from scripts/benchmark_performance.py (the run to analyze / the adaptive run)')
    p.add_argument('--baseline', default=None,
                   help='report from a run with ADAPTIVE_ROUTING_ENABLED=false, to compare against')
    p.add_argument('--output', default=None, help='also write the analysis as JSON')
    args = p.parse_args(argv)

    try:
        run = load_run(args.run)
        baseline = load_run(args.baseline) if args.baseline else None
    except (OSError, ValueError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2

    analysis = analyze_run(run)
    print(format_by_path(analysis))
    report: Dict = {'schema_version': 1, 'analysis': analysis}
    if baseline is not None:
        comparison = compare_runs(baseline, run)
        report['baseline_label'] = baseline.get('label')
        report['comparison'] = comparison
        print(format_comparison(comparison))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f'\nWrote {args.output}')
    return 0 if analysis['usable_records'] else 1


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
