"""Build the `performance` response block and the structured log line.

Only genuinely measured fields are emitted: a stage that did not run, tokens the
provider did not report, or a cost that cannot be computed are OMITTED, never
zero-filled.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from performance.pricing import PricingConfig
from performance.recorder import STAGE_ORDER, RequestRecorder

ROUTING_PATH_FULL_RETRIEVAL = 'full_retrieval'   # ADAPTIVE_ROUTING_ENABLED=false (legacy path)

LOGGER_NAME = 'enterprise_rag.performance'
logger = logging.getLogger(LOGGER_NAME)


def routing_path_from_adaptive(result) -> Optional[str]:
    """The path that produced the final chunks: 'direct', 'decomposed' or
    'reformulated' (one retry happened; the original route stays in
    ``routing.path``). Duck-typed on AdaptiveResult."""
    attempts = getattr(result, 'attempts', None)
    if attempts:
        return getattr(attempts[-1], 'kind', None)
    decision = getattr(result, 'decision', None)
    return getattr(decision, 'path', None)


def build_performance(recorder: RequestRecorder, *, routing_path: Optional[str] = None,
                      pricing: Optional[PricingConfig] = None) -> Dict:
    perf: Dict = {'total_latency_ms': round(recorder.elapsed_ms(), 2)}

    measured = recorder.stages_ms
    ordered = [n for n in STAGE_ORDER if n in measured] + [n for n in measured if n not in STAGE_ORDER]
    if ordered:
        perf['stages'] = {f'{name}_ms': round(measured[name], 2) for name in ordered}

    tokens = recorder.token_summary()
    if tokens is not None:
        perf['tokens'] = tokens
        if pricing is not None:
            cost = pricing.estimate_cost(tokens['input'], tokens['output'])
            if cost is not None:
                perf['estimated_cost'] = cost
                perf['cost_currency'] = pricing.currency

    if routing_path:
        perf['routing_path'] = routing_path
    return perf


def log_performance(perf: Dict, *, refused: Optional[bool] = None) -> None:
    """One JSON line on logger 'enterprise_rag.performance' (INFO). Contains
    timings, token counts, cost and the routing path ONLY: never the question,
    the answer, retrieved text or the session id. Silent unless the operator
    enables INFO logging for this logger."""
    if not logger.isEnabledFor(logging.INFO):
        return
    record = {'event': 'rag_query_performance', **perf}
    if refused is not None:
        record['refused'] = refused
    logger.info(json.dumps(record, sort_keys=True))
