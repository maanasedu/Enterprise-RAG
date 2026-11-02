"""Per-request stage timing and LLM-usage collection.

The pipeline is instrumented with ``with stage('dense_retrieval'): ...`` and
``record_llm_usage('generation', response)``. Both are no-ops unless a
``RequestRecorder`` has been activated for the current request, so retrieval
code behaves identically in scripts, evaluation and tests that never activate
one. Recording can never raise into the pipeline.

Stage semantics
  * A stage that runs several times in one request (dense retrieval once per
    sub-query, reranking again on the reformulation retry) is SUMMED.
  * Stages are independent measurements, not a partition of the total:
    ``rerank_throttle_wait`` happens inside ``reranking``, and work that is not
    a named stage (guards, evidence evaluation, prompt building) is only in the
    total.
"""
from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from performance.tokens import TokenUsage, aggregate_usage, extract_usage

# Canonical stage names; the response key is f'{name}_ms'.
DENSE_RETRIEVAL = 'dense_retrieval'
BM25_RETRIEVAL = 'bm25_retrieval'
FUSION = 'fusion'
RERANKING = 'reranking'
RERANK_THROTTLE_WAIT = 'rerank_throttle_wait'   # subset of RERANKING
ROUTING = 'routing'
REFORMULATION = 'reformulation'
GENERATION = 'generation'

STAGE_ORDER = (
    DENSE_RETRIEVAL, BM25_RETRIEVAL, FUSION, RERANKING, RERANK_THROTTLE_WAIT,
    ROUTING, REFORMULATION, GENERATION,
)

PURPOSE_GENERATION = 'generation'
PURPOSE_REFORMULATION = 'reformulation'


class RequestRecorder:
    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._start = clock()
        self._stages_ms: Dict[str, float] = {}
        self._stage_calls: Dict[str, int] = {}
        self._llm_calls: List[Tuple[str, Optional[TokenUsage]]] = []

    # ── timing ──────────────────────────────────────────────────────────
    def add_stage(self, name: str, elapsed_ms: float) -> None:
        self._stages_ms[name] = self._stages_ms.get(name, 0.0) + max(0.0, elapsed_ms)
        self._stage_calls[name] = self._stage_calls.get(name, 0) + 1

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = self._clock()
        try:
            yield
        finally:
            self.add_stage(name, (self._clock() - started) * 1000.0)

    def elapsed_ms(self) -> float:
        return (self._clock() - self._start) * 1000.0

    @property
    def stages_ms(self) -> Dict[str, float]:
        return dict(self._stages_ms)

    def stage_calls(self, name: str) -> int:
        return self._stage_calls.get(name, 0)

    # ── LLM usage ───────────────────────────────────────────────────────
    def record_llm_call(self, purpose: str, usage: Optional[TokenUsage]) -> None:
        self._llm_calls.append((purpose, usage))

    @property
    def llm_calls(self) -> List[Tuple[str, Optional[TokenUsage]]]:
        return list(self._llm_calls)

    def token_summary(self) -> Optional[Dict]:
        return aggregate_usage(self._llm_calls)


# ── request-scoped access ───────────────────────────────────────────────────
_current: ContextVar[Optional[RequestRecorder]] = ContextVar('rag_request_recorder', default=None)
_NOOP = nullcontext()


def activate(recorder: RequestRecorder) -> Token:
    return _current.set(recorder)


def deactivate(token: Token) -> None:
    _current.reset(token)


def current_recorder() -> Optional[RequestRecorder]:
    return _current.get()


def stage(name: str):
    """Time the enclosed block under `name` (no-op without an active recorder)."""
    recorder = _current.get()
    return _NOOP if recorder is None else recorder.stage(name)


def record_llm_usage(purpose: str, response) -> None:
    """Record that one LLM call happened and what usage the provider reported.
    Never raises: a malformed response must not break generation."""
    recorder = _current.get()
    if recorder is None:
        return
    try:
        recorder.record_llm_call(purpose, extract_usage(response))
    except Exception:  # noqa: BLE001 - instrumentation must never break the pipeline
        recorder.record_llm_call(purpose, None)
