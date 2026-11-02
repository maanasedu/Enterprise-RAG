"""Adaptive retrieval orchestration (Dimension 2).

    question
      -> QueryRouter                      (router.py, deterministic)
      -> LOW : hybrid.retrieve(question)                       [direct]
         HIGH: retrieve_candidates per sub-query -> merge (RRF)
               -> rerank once against the ORIGINAL question    [decomposed]
      -> EvidenceEvaluator                (evidence.py, thresholds)
      -> weak? ONE reformulated cycle     (reformulator.py)
              retrieve -> rerank -> evaluate again
      -> AdaptiveResult (sufficient | refused)

This module only wires those parts together and reports what happened. It
does NOT generate: the caller (the API) decides to generate or to return the
structured refusal based on `AdaptiveResult.refused`, which keeps routing,
evidence scoring, thresholds and generation independently changeable.

Reranking always scores candidates against the user's original question,
even when candidates were retrieved with sub-queries or rewrites, so the
evidence score answers "how well does this chunk match what was asked".

The retry is straight-line code, not a loop: there is at most one
reformulation cycle, i.e. at most two retrieve+rerank rounds (two Cohere
calls) per question, by construction.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from performance.recorder import REFORMULATION, ROUTING, stage
from retrieval.env_config import parse_bool, parse_int
from retrieval.evidence import EvidenceConfig, EvidenceEvaluator, EvidenceVerdict
from retrieval.reformulator import HeuristicReformulator, LLMReformulator
from retrieval.router import (
    Complexity,
    QueryRouter,
    RouterConfig,
    RoutingDecision,
    normalize_query,
)

REFUSAL_MESSAGE = 'Insufficient evidence to answer accurately.'
REFUSAL_REASON = 'insufficient_evidence'

ENV_ENABLED = 'ADAPTIVE_ROUTING_ENABLED'
ENV_REFORMULATION_MODE = 'ADAPTIVE_REFORMULATION_MODE'
ENV_MAX_SUBQUERIES = 'ADAPTIVE_MAX_SUBQUERIES'
ENV_LONG_QUERY_WORDS = 'ADAPTIVE_LONG_QUERY_WORDS'

REFORMULATION_MODES = ('heuristic', 'llm')

RETRY_SKIPPED_NO_DISTINCT_QUERY = 'no_distinct_reformulation'


@dataclass(frozen=True)
class AdaptiveConfig:
    enabled: bool = True
    reformulation_mode: str = 'heuristic'
    router: RouterConfig = RouterConfig()
    evidence: EvidenceConfig = EvidenceConfig()

    def __post_init__(self):
        if self.reformulation_mode not in REFORMULATION_MODES:
            raise ValueError(
                f'reformulation_mode must be one of {REFORMULATION_MODES}, '
                f'got {self.reformulation_mode!r}'
            )

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> 'AdaptiveConfig':
        env = os.environ if env is None else env
        defaults = RouterConfig()
        mode = (env.get(ENV_REFORMULATION_MODE) or 'heuristic').strip().lower()
        return cls(
            enabled=parse_bool(ENV_ENABLED, env.get(ENV_ENABLED), True),
            reformulation_mode=mode,
            router=RouterConfig(
                long_query_words=parse_int(
                    ENV_LONG_QUERY_WORDS, env.get(ENV_LONG_QUERY_WORDS),
                    defaults.long_query_words),
                max_subqueries=parse_int(
                    ENV_MAX_SUBQUERIES, env.get(ENV_MAX_SUBQUERIES),
                    defaults.max_subqueries),
            ),
            evidence=EvidenceConfig.from_env(env),
        )


@dataclass(frozen=True)
class Attempt:
    """One retrieve -> rerank -> evaluate round."""
    kind: str                      # 'direct' | 'decomposed' | 'reformulated'
    queries: Tuple[str, ...]       # queries retrieved with in this round
    verdict: EvidenceVerdict

    def to_dict(self) -> Dict:
        return {
            'kind': self.kind,
            'queries': list(self.queries),
            'sufficient': self.verdict.sufficient,
            'reason': self.verdict.reason,
            'top_score': self.verdict.top_score,
        }


@dataclass
class AdaptiveResult:
    # Final reranked chunks. When `refused` is True these are the weak
    # chunks that failed evaluation and must NOT be used for generation;
    # they are kept only so callers can log/inspect them.
    chunks: List[Dict]
    decision: RoutingDecision
    attempts: List[Attempt]
    retry_skipped_reason: Optional[str] = None

    @property
    def verdict(self) -> EvidenceVerdict:
        return self.attempts[-1].verdict

    @property
    def sufficient(self) -> bool:
        return self.verdict.sufficient

    @property
    def refused(self) -> bool:
        return not self.sufficient

    @property
    def retried(self) -> bool:
        return len(self.attempts) > 1

    def routing_dict(self) -> Dict:
        return self.decision.to_dict()

    def evidence_dict(self) -> Dict:
        return {
            **self.verdict.to_dict(),
            'retried': self.retried,
            'retry_skipped_reason': self.retry_skipped_reason,
            'attempts': [a.to_dict() for a in self.attempts],
        }


class AdaptiveRetriever:
    def __init__(self, hybrid, router: Optional[QueryRouter] = None,
                 evaluator: Optional[EvidenceEvaluator] = None,
                 reformulator=None):
        self.hybrid = hybrid
        self.router = router or QueryRouter()
        self.evaluator = evaluator or EvidenceEvaluator()
        self.reformulator = reformulator or HeuristicReformulator()

    def retrieve(self, question: str, top_k_retrieve: int = 50,
                 top_k_rerank: int = 5) -> AdaptiveResult:
        decision = self._route(question)
        used_queries = [question] + list(decision.subqueries)

        # ── Round 1 ─────────────────────────────────────────────────────
        if decision.decomposed:
            kind, queries = 'decomposed', tuple(decision.subqueries)
            chunks = self._retrieve_many(question, queries, top_k_retrieve, top_k_rerank)
        else:
            # Exactly the pre-existing retrieval call.
            kind, queries = 'direct', (question,)
            chunks = self.hybrid.retrieve(
                question, top_k_retrieve=top_k_retrieve, top_k_rerank=top_k_rerank
            )

        verdict = self.evaluator.evaluate(chunks)
        attempts = [Attempt(kind, queries, verdict)]
        if verdict.sufficient or not verdict.retryable:
            return AdaptiveResult(chunks, decision, attempts)

        # ── Round 2: the single allowed reformulation cycle ─────────────
        new_queries = self._reformulate(question, used_queries, decision)
        if not new_queries:
            return AdaptiveResult(
                chunks, decision, attempts,
                retry_skipped_reason=RETRY_SKIPPED_NO_DISTINCT_QUERY,
            )

        chunks = self._retrieve_many(question, new_queries, top_k_retrieve, top_k_rerank)
        attempts.append(
            Attempt('reformulated', tuple(new_queries), self.evaluator.evaluate(chunks))
        )
        return AdaptiveResult(chunks, decision, attempts)

    # ── helpers ─────────────────────────────────────────────────────────
    def _route(self, question: str) -> RoutingDecision:
        try:
            with stage(ROUTING):
                return self.router.route(question)
        except Exception as e:
            # A routing bug must not take down /query; degrade to the
            # plain direct-retrieval behavior.
            print(f'Query routing failed ({e}) — falling back to direct retrieval')
            return RoutingDecision(Complexity.LOW, ('router_error',), (question,))

    def _reformulate(self, question: str, used_queries: Sequence[str],
                     decision: RoutingDecision) -> List[str]:
        try:
            with stage(REFORMULATION):
                proposed = self.reformulator.reformulate(question, used_queries, decision)
        except Exception as e:
            print(f'Query reformulation failed ({e}) — skipping retry')
            return []

        # Enforce the contract here too, whatever the reformulator returned.
        seen = {normalize_query(q) for q in used_queries}
        out: List[str] = []
        for q in proposed or []:
            key = normalize_query(q) if isinstance(q, str) else ''
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(q)
            if len(out) >= self.router.config.max_subqueries:
                break
        return out

    def _retrieve_many(self, question: str, queries: Sequence[str],
                       top_k_retrieve: int, top_k_rerank: int) -> List[Dict]:
        """Retrieve candidates for every query (no reranking), merge them
        into one pool, then rerank ONCE against the original question."""
        pools = [self.hybrid.retrieve_candidates(q, top_k_retrieve) for q in queries]
        merged = self.hybrid.merge_candidate_pools(pools, limit=top_k_retrieve)
        return self.hybrid.rerank_candidates(question, merged, top_k_rerank)


def build_adaptive_retriever(hybrid, config: AdaptiveConfig, generator=None) -> AdaptiveRetriever:
    """Assemble an AdaptiveRetriever from config. `generator` (the existing
    RAGGenerator) is only used to reach its LLM client when
    reformulation_mode == 'llm'."""
    client = getattr(generator, 'client', None)
    model = getattr(generator, 'model', None)
    if config.reformulation_mode == 'llm' and client is not None and model:
        reformulator = LLMReformulator(client, model)
    else:
        if config.reformulation_mode == 'llm':
            print('ADAPTIVE_REFORMULATION_MODE=llm but no LLM client is available '
                  '— using heuristic reformulation')
        reformulator = HeuristicReformulator()
    return AdaptiveRetriever(
        hybrid,
        router=QueryRouter(config.router),
        evaluator=EvidenceEvaluator(config.evidence),
        reformulator=reformulator,
    )
