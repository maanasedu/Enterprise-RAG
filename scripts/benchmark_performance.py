"""D5-Lite -- lightweight latency / throughput benchmark for a running API.

Sends existing Dimension 1 benchmark questions to POST /query (with
``include_performance: true``) at a configurable concurrency and reports:

  total / successful / failed requests, throughput (RPS),
  average / P50 / P95 / P99 latency, and a latency breakdown by pipeline stage.

    # terminal 1: start the API as usual (uvicorn / app.py)
    # terminal 2:
    python scripts/benchmark_performance.py --label adaptive --limit 20

This is deliberately NOT a load-testing platform: no ramp-up, no distributed
workers, no think-time model. It answers "how long does this pipeline take on
real benchmark questions, and where does the time go?".

Questions are never written by hand: they are loaded from the existing
benchmark splits (data/evaluation/{calibration,test}.json) through the
Dimension 1 loader, in file order, so a given --split/--limit/--categories
always selects the same questions. The held-out test split needs
--confirm-final-test, exactly as in the Dimension 1 evaluation.

Latency here is CLIENT-observed wall time (includes network and any server-side
queueing). Stage timings and tokens come from the server's own ``performance``
block. The report stores ids, categories, timings, token counts and cost only:
no question text and no answer text.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.retrieval_eval import (          # Dimension 1 -- reused, not reimplemented
    BenchmarkQuestion,
    RetrievalEvalError,
    guard_split,
    load_questions,
    split_path,
)
from performance.stats import sample_size_warnings, summarize

DEFAULT_BENCHMARK_DIR = os.path.join('data', 'evaluation')
DEFAULT_OUTPUT_DIR = os.path.join('data', 'performance')
DEFAULT_SPLIT = 'calibration'      # the small split; test is held out (see guard_split)
DEFAULT_LIMIT = 20                 # small on purpose: every request spends LLM tokens
DEFAULT_TOP_K = 5                  # api/rag_api.py's QueryRequest default
DEFAULT_TIMEOUT_S = 120.0


# ── data ─────────────────────────────────────────────────────────────────────
@dataclass
class RequestResult:
    index: int
    id: str
    category: str
    ok: bool
    client_latency_ms: float
    status_code: Optional[int] = None
    error: Optional[str] = None          # exception class / HTTP status only, never a response body
    refused: Optional[bool] = None
    performance: Optional[Dict] = None   # the server's `performance` block, when it sent one

    def to_dict(self) -> Dict:
        return {
            'id': self.id, 'category': self.category, 'ok': self.ok,
            'status_code': self.status_code, 'error': self.error,
            'client_latency_ms': round(self.client_latency_ms, 2),
            'refused': self.refused, 'performance': self.performance,
        }


@dataclass
class BenchmarkRun:
    results: List[RequestResult]
    wall_seconds: float
    warmup_requests: int = 0


Sender = Callable[[int, BenchmarkQuestion], RequestResult]


# ── question selection ───────────────────────────────────────────────────────
def select_questions(questions: Sequence[BenchmarkQuestion], *, limit: Optional[int] = None,
                     categories: Optional[Sequence[str]] = None,
                     exclude_categories: Optional[Sequence[str]] = None) -> List[BenchmarkQuestion]:
    """Deterministic subset: filter by category, keep file order, take the first `limit`."""
    selected = list(questions)
    if categories:
        keep = set(categories)
        selected = [q for q in selected if q.category in keep]
    if exclude_categories:
        drop = set(exclude_categories)
        selected = [q for q in selected if q.category not in drop]
    if limit is not None:
        if limit < 0:
            raise ValueError('limit must be >= 0')
        selected = selected[:limit]
    return selected


# ── sending ──────────────────────────────────────────────────────────────────
def make_http_sender(base_url: str, *, top_k: int, timeout_s: float, run_id: str, client=None) -> Sender:
    """A sender that POSTs one question to {base_url}/query.

    Every request gets its own session_id: the API keeps per-session
    conversation memory, and reusing one session would make later prompts
    longer (history) and change the context budget, contaminating tokens/latency.
    """
    import httpx

    http = client or httpx.Client(timeout=timeout_s)
    url = base_url.rstrip('/') + '/query'

    def send(index: int, q: BenchmarkQuestion) -> RequestResult:
        base = dict(index=index, id=q.id, category=q.category)
        payload = {'question': q.question, 'top_k': top_k, 'include_performance': True,
                   'session_id': f'perf-{run_id}-{index}'}
        started = time.perf_counter()
        try:
            response = http.post(url, json=payload)
        except Exception as e:  # noqa: BLE001 - a failed request is a data point, not a crash
            return RequestResult(**base, ok=False, error=type(e).__name__,
                                 client_latency_ms=(time.perf_counter() - started) * 1000)
        latency_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            return RequestResult(**base, ok=False, status_code=response.status_code,
                                 error=f'HTTP {response.status_code}', client_latency_ms=latency_ms)
        try:
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError('response is not a JSON object')
        except ValueError:
            return RequestResult(**base, ok=False, status_code=200, error='invalid JSON response',
                                 client_latency_ms=latency_ms)
        perf = body.get('performance')
        return RequestResult(**base, ok=True, status_code=200, client_latency_ms=latency_ms,
                             refused=body.get('refused'),
                             performance=perf if isinstance(perf, dict) else None)

    return send


def cleanup_sessions(base_url: str, run_id: str, count: int, timeout_s: float = 10.0) -> None:
    """Best-effort: drop the per-request conversation sessions this run created."""
    import httpx
    url = base_url.rstrip('/') + '/clear_session'
    try:
        with httpx.Client(timeout=timeout_s) as http:
            for i in range(count):
                http.post(url, json={'session_id': f'perf-{run_id}-{i}'})
    except Exception:  # noqa: BLE001 - cleanup must never fail the run
        pass


# ── running ──────────────────────────────────────────────────────────────────
def run_benchmark(questions: Sequence[BenchmarkQuestion], send: Sender, *, concurrency: int = 1,
                  warmup: int = 0, clock: Callable[[], float] = time.perf_counter) -> BenchmarkRun:
    """Run every question once through `send`.

    `send` must return a RequestResult; if it raises anyway, that request is
    recorded as failed rather than aborting the run. Results are returned in
    question order whatever the completion order was. Warm-up requests (the
    first `warmup` questions, run sequentially) are sent but not counted.
    """
    if concurrency < 1:
        raise ValueError('concurrency must be >= 1')
    if warmup < 0:
        raise ValueError('warmup must be >= 0')

    def safe_send(i: int, q: BenchmarkQuestion) -> RequestResult:
        try:
            return send(i, q)
        except Exception as e:  # noqa: BLE001
            return RequestResult(index=i, id=q.id, category=q.category, ok=False,
                                 error=type(e).__name__, client_latency_ms=0.0)

    warmed = 0
    for i, q in enumerate(list(questions)[:warmup]):
        safe_send(-1 - i, q)
        warmed += 1

    started = clock()
    if concurrency == 1:
        results = [safe_send(i, q) for i, q in enumerate(questions)]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda iq: safe_send(*iq), enumerate(questions)))
    wall = clock() - started
    return BenchmarkRun(results=sorted(results, key=lambda r: r.index), wall_seconds=wall,
                        warmup_requests=warmed)


# ── summarizing ──────────────────────────────────────────────────────────────
def stage_breakdown(results: Sequence[RequestResult]) -> Dict[str, Dict]:
    """Per-stage statistics over the successful responses that reported that
    stage. `n` is how many reported it (a stage that did not run for a request
    is absent for it, not zero)."""
    by_stage: Dict[str, List[float]] = {}
    for r in results:
        if not (r.ok and r.performance):
            continue
        for name, ms in (r.performance.get('stages') or {}).items():
            if isinstance(ms, (int, float)) and not isinstance(ms, bool):
                by_stage.setdefault(name, []).append(float(ms))
    return {name: summarize(values) for name, values in sorted(by_stage.items())}


def summarize_run(run: BenchmarkRun) -> Dict:
    results = run.results
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    with_perf = [r for r in ok if r.performance]
    server_totals = [r.performance['total_latency_ms'] for r in with_perf
                     if isinstance(r.performance.get('total_latency_ms'), (int, float))]

    errors: Dict[str, int] = {}
    for r in failed:
        errors[r.error or 'unknown'] = errors.get(r.error or 'unknown', 0) + 1

    paths: Dict[str, int] = {}
    for r in with_perf:
        path = r.performance.get('routing_path') or 'no_retrieval'
        paths[path] = paths.get(path, 0) + 1

    wall = run.wall_seconds
    return {
        'total_requests': len(results),
        'successful_requests': len(ok),
        'failed_requests': len(failed),
        'refused_responses': sum(1 for r in ok if r.refused),
        'wall_clock_seconds': round(wall, 3),
        # successful requests per wall-clock second, from first send to last completion
        'throughput_rps': round(len(ok) / wall, 4) if wall > 0 else None,
        'client_latency_ms': summarize([r.client_latency_ms for r in ok]),
        'server_total_latency_ms': summarize(server_totals),
        'stage_latency_ms': stage_breakdown(results),
        'routing_paths': dict(sorted(paths.items())),
        'responses_without_performance_block': len(ok) - len(with_perf),
        'errors': dict(sorted(errors.items())),
        'warnings': sample_size_warnings(len(ok)),
    }


def redact_url(url: str) -> str:
    """Drop any credentials/query from the URL before it is written to disk."""
    parts = urlsplit(url)
    host = parts.hostname or ''
    netloc = f'{host}:{parts.port}' if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, '', ''))


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='D5-Lite: latency/throughput benchmark of a running /query API on '
                    'existing Dimension 1 benchmark questions.')
    p.add_argument('--base-url', default='http://localhost:8000', help='API base URL (default %(default)s)')
    p.add_argument('--benchmark-dir', default=DEFAULT_BENCHMARK_DIR)
    p.add_argument('--split', default=DEFAULT_SPLIT, choices=('calibration', 'test'),
                   help='benchmark split to draw questions from (default %(default)s)')
    p.add_argument('--confirm-final-test', action='store_true',
                   help='required for --split test (held out; see Dimension 1)')
    p.add_argument('--limit', type=int, default=DEFAULT_LIMIT,
                   help='use the first N questions after filtering (default %(default)s; every request spends LLM tokens)')
    p.add_argument('--categories', default=None, help='comma-separated categories to include')
    p.add_argument('--exclude-categories', default=None,
                   help='comma-separated categories to skip (e.g. adversarial: those are refused by the input guard, so they exercise no retrieval or generation)')
    p.add_argument('--concurrency', type=int, default=1, help='parallel client requests (default %(default)s)')
    p.add_argument('--warmup', type=int, default=0, help='uncounted requests sent first (default %(default)s)')
    p.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    p.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT_S, help='per-request timeout, seconds')
    p.add_argument('--label', default='run', help='names this run; used in the default output filename')
    p.add_argument('--output', default=None, help='report path (default data/performance/benchmark_<label>.json)')
    return p


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    return [v.strip() for v in value.split(',') if v.strip()] if value else None


def _print_summary(summary: Dict, output: str) -> None:
    lat = summary['client_latency_ms']
    srv = summary['server_total_latency_ms']
    print()
    print(f"Requests: total={summary['total_requests']}  ok={summary['successful_requests']}  "
          f"failed={summary['failed_requests']}  refused={summary['refused_responses']}")
    print(f"Wall clock: {summary['wall_clock_seconds']} s   throughput: {summary['throughput_rps']} req/s")
    for title, stats in (('Client latency (ms)', lat), ('Server total latency (ms)', srv)):
        print(f"{title}: n={stats['n']} mean={stats['mean']} p50={stats['p50']} "
              f"p95={stats['p95']} p99={stats['p99']} max={stats['max']}")
    if summary['stage_latency_ms']:
        print('Stage latency (ms), over responses that ran the stage:')
        for name, st in summary['stage_latency_ms'].items():
            print(f"  {name:<24} n={st['n']:<4} mean={st['mean']:<10} p50={st['p50']:<10} p95={st['p95']}")
    if summary['routing_paths']:
        print(f"Routing paths: {summary['routing_paths']}")
    if summary['errors']:
        print(f"Errors: {summary['errors']}")
    if summary['responses_without_performance_block']:
        print(f"NOTE: {summary['responses_without_performance_block']} successful response(s) had no "
              f"`performance` block (is the server running the D5-Lite code?)")
    for w in summary['warnings']:
        print(f'NOTE: {w}')
    print(f'\nFull report: {output}')


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        guard_split(args.split, args.confirm_final_test)
        path = split_path(args.benchmark_dir, args.split)
        all_questions = load_questions(path)
        questions = select_questions(
            all_questions, limit=args.limit, categories=_split_csv(args.categories),
            exclude_categories=_split_csv(args.exclude_categories))
        if not questions:
            raise RetrievalEvalError('no questions selected (check --split/--categories/--limit)')
    except (RetrievalEvalError, ValueError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2

    import httpx
    try:
        httpx.get(args.base_url.rstrip('/') + '/health', timeout=10).raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f'ERROR: cannot reach {redact_url(args.base_url)}/health ({type(e).__name__}). '
              f'Start the API first.', file=sys.stderr)
        return 2

    run_id = uuid.uuid4().hex[:8]
    print(f'Benchmarking {len(questions)} question(s) from {path} '
          f'(concurrency={args.concurrency}, warmup={args.warmup})')
    send = make_http_sender(args.base_url, top_k=args.top_k, timeout_s=args.timeout, run_id=run_id)
    try:
        run = run_benchmark(questions, send, concurrency=args.concurrency, warmup=args.warmup)
    finally:
        cleanup_sessions(args.base_url, run_id, len(questions) + args.warmup)

    summary = summarize_run(run)
    output = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f'benchmark_{args.label}.json')
    report = {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'dimension': 'D5-Lite performance benchmark',
        'label': args.label,
        'protocol': {
            'base_url': redact_url(args.base_url), 'split': args.split, 'benchmark_file': path,
            'n_questions': len(questions), 'limit': args.limit,
            'categories': _split_csv(args.categories), 'exclude_categories': _split_csv(args.exclude_categories),
            'concurrency': args.concurrency, 'warmup': run.warmup_requests, 'top_k': args.top_k,
            'unique_session_per_request': True,
            'latency_definition': 'client-observed wall time per request; stage timings are server-reported',
        },
        'environment': {'python': platform.python_version(), 'platform': platform.platform()},
        'summary': summary,
        'per_request': [r.to_dict() for r in run.results],
    }
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    _print_summary(summary, output)
    return 0 if summary['successful_requests'] > 0 else 1


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
