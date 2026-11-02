"""Dimension 2 -- held-out evaluation of the adaptive-routing pipeline.

Runs the untouched 202-question TEST split (data/evaluation/test.json)
through the *existing, unmodified* production pipeline -- the same
QueryRouter -> HybridRetriever -> CohereReranker -> EvidenceEvaluator ->
(reformulate once if weak) -> refuse-or-not decision that
retrieval/adaptive.py::AdaptiveRetriever makes for every /query call --
and reports routing, evidence and refusal statistics.

This intentionally stops before generation. The refusal decision Dimension
2 makes is fully determined by AdaptiveRetriever.retrieve() -- api/rag_api.py
only calls the LLM generator afterward, and only when NOT refused. So
running generation (or RAGAS) here would not change a single refused/
answered outcome; it would only add an LLM call and cost per question with
no effect on the numbers this script reports. Hence: no generation call.

    python scripts/evaluate_adaptive_rag.py

Requires:
  * COHERE_API_KEY (real reranker scores; see MissingCredentialsError)
  * The real FAISS/BM25 corpus in data/processed/ (not Git LFS pointers --
    run `git lfs pull` first if needed; see IndexNotAvailableError)

Does NOT modify data/evaluation/test.json, any production retrieval code,
the calibrated evidence threshold, or existing tests. Does not call the
LLM generator or run RAGAS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from evaluation.retrieval_eval import (          # Dimension 1 -- reused, not reimplemented
    MissingCredentialsError,
    RetrievalEvalError,
    assert_gold_in_corpus,
    build_retrievers,
    load_production_components,
    load_questions,
    split_path,
)
from retrieval.adaptive import AdaptiveConfig, build_adaptive_retriever

DEFAULT_BENCHMARK_DIR = os.path.join('data', 'evaluation')
DEFAULT_PROCESSED_DIR = os.path.join('data', 'processed')
DEFAULT_OUTPUT = os.path.join('data', 'evaluation', 'adaptive_rag_test.json')
DEFAULT_SPLIT = 'test'

# Production /query defaults (api/rag_api.py's QueryRequest / hybrid.retrieve calls).
DEFAULT_TOP_K_RERANK = 5
DEFAULT_TOP_K_RETRIEVE = 50

ADVERSARIAL_CATEGORY = 'adversarial'


@dataclass
class QuestionResult:
    id: str
    category: str
    evidence_status: str
    answerable: bool
    error: Optional[str] = None
    # Populated only when `error` is None.
    routing_complexity: Optional[str] = None
    routing_path: Optional[str] = None
    routing_signals: Optional[List[str]] = None
    n_subqueries: Optional[int] = None
    evidence_evaluated: Optional[bool] = None
    round1_sufficient: Optional[bool] = None      # "weak evidence" iff False
    top_score: Optional[float] = None
    final_reason: Optional[str] = None
    retried: Optional[bool] = None
    retry_skipped_reason: Optional[str] = None
    refused: Optional[bool] = None

    @property
    def is_adversarial(self) -> bool:
        return self.category == ADVERSARIAL_CATEGORY

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'category': self.category,
            'evidence_status': self.evidence_status,
            'answerable': self.answerable,
            'error': self.error,
            'routing': {
                'complexity': self.routing_complexity,
                'path': self.routing_path,
                'signals': self.routing_signals,
                'n_subqueries': self.n_subqueries,
            } if self.error is None else None,
            'evidence': {
                'evaluated': self.evidence_evaluated,
                'round1_sufficient': self.round1_sufficient,
                'top_score': self.top_score,
                'final_reason': self.final_reason,
            } if self.error is None else None,
            'retried': self.retried,
            'retry_skipped_reason': self.retry_skipped_reason,
            'refused': self.refused,
        }


def evaluate_question(q, retriever, *, top_k_retrieve: int, top_k_rerank: int) -> QuestionResult:
    """Run one question through the real AdaptiveRetriever. Any exception
    is caught here (not left to crash the 202-question run) and recorded
    as an error result instead."""
    base = dict(id=q.id, category=q.category, evidence_status=q.evidence_status,
               answerable=q.answerable)
    try:
        result = retriever.retrieve(q.question, top_k_retrieve=top_k_retrieve,
                                    top_k_rerank=top_k_rerank)
    except Exception as e:  # noqa: BLE001 -- deliberately broad: this is a harness
        return QuestionResult(**base, error=f'{type(e).__name__}: {e}')

    round1 = result.attempts[0].verdict
    return QuestionResult(
        **base,
        routing_complexity=result.decision.complexity.value,
        routing_path=result.decision.path,
        routing_signals=list(result.decision.signals),
        n_subqueries=len(result.decision.subqueries),
        evidence_evaluated=result.verdict.evaluated,
        round1_sufficient=round1.sufficient,
        top_score=result.verdict.top_score,
        final_reason=result.verdict.reason,
        retried=result.retried,
        retry_skipped_reason=result.retry_skipped_reason,
        refused=result.refused,
    )


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def summarize(results: Sequence[QuestionResult]) -> Dict:
    ok = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    sufficient = [r for r in ok if r.evidence_status == 'sufficient']
    insufficient = [r for r in ok if r.evidence_status == 'insufficient']
    ambiguous = [r for r in ok if r.evidence_status == 'ambiguous']
    adversarial = [r for r in ok if r.is_adversarial]

    n_direct = sum(1 for r in ok if r.routing_complexity == 'low')
    n_complex = sum(1 for r in ok if r.routing_complexity == 'high')
    n_decomposed = sum(1 for r in ok if r.routing_path == 'decomposed')

    n_evaluated = sum(1 for r in ok if r.evidence_evaluated)
    n_weak = sum(1 for r in ok if r.round1_sufficient is False)
    n_reformulated = sum(1 for r in ok if r.retried)
    n_refused = sum(1 for r in ok if r.refused)

    refused_insufficient = sum(1 for r in insufficient if r.refused)
    refused_sufficient = sum(1 for r in sufficient if r.refused)

    return {
        'total_questions': len(results),
        'answerable_count': sum(1 for r in ok if r.answerable),
        'insufficient_count': len(insufficient),
        'ambiguous_count': len(ambiguous),
        'adversarial_count': len(adversarial),
        'routing': {
            'direct': n_direct,
            'complex': n_complex,
            'complex_decomposed': n_decomposed,
            'complex_not_decomposable': n_complex - n_decomposed,
        },
        'evidence': {
            'evaluated': n_evaluated,
            'not_evaluated_fallback_scores': len(ok) - n_evaluated,
            'weak_evidence_on_first_attempt': n_weak,
        },
        'reformulations': n_reformulated,
        'refusals': n_refused,
        'refusal_rate_on_insufficient': _rate(refused_insufficient, len(insufficient)),
        'false_refusal_rate_on_sufficient': _rate(refused_sufficient, len(sufficient)),
        'errors': len(errors),
        'by_evidence_status': {
            status: {
                'n': len(group),
                'refused': sum(1 for r in group if r.refused),
            }
            for status, group in (
                ('sufficient', sufficient), ('insufficient', insufficient),
                ('ambiguous', ambiguous),
            )
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Dimension 2 -- evaluate the adaptive-routing pipeline on '
                    'the held-out test split (production retrieval/evidence code).')
    p.add_argument('--benchmark-dir', default=DEFAULT_BENCHMARK_DIR)
    p.add_argument('--processed-dir', default=DEFAULT_PROCESSED_DIR)
    p.add_argument('--split', default=DEFAULT_SPLIT,
                   help='Which benchmark split to evaluate (default: test, the '
                       '202-question held-out set).')
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--top-k-rerank', type=int, default=DEFAULT_TOP_K_RERANK)
    p.add_argument('--top-k-retrieve', type=int, default=DEFAULT_TOP_K_RETRIEVE)
    p.add_argument('--limit', type=int, default=None,
                   help='Smoke-test only: evaluate the first N questions.')
    return p


def _progress_factory():
    try:
        from tqdm import tqdm
        return lambda it, desc=None: tqdm(it, desc=desc)
    except Exception:  # pragma: no cover - tqdm is optional
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if not os.getenv('COHERE_API_KEY', ''):
            raise MissingCredentialsError(
                'COHERE_API_KEY is not set. This evaluation needs real Cohere '
                'reranker scores -- a fallback score is not a real evaluation '
                'of the evidence gate (see retrieval/reranker.py). Set it and re-run.'
            )

        path = split_path(args.benchmark_dir, args.split)
        questions = load_questions(path)
        if args.limit is not None:
            questions = questions[:args.limit]
        print(f'Loaded {len(questions)} question(s) from {path}')

        store, bm25, embedder = load_production_components(args.processed_dir)
        assert_gold_in_corpus(questions, store.metadata)

        hybrid = build_retrievers(store, bm25, embedder, configs=['hybrid_cohere'])['hybrid_cohere']

        # Exactly how api/rag_api.py builds the adaptive retriever for
        # /query: production AdaptiveConfig (calibrated threshold included,
        # via retrieval/evidence.py's default), no generator (no LLM call).
        adaptive_config = AdaptiveConfig.from_env()
        retriever = build_adaptive_retriever(hybrid, adaptive_config, generator=None)

        progress = _progress_factory()
        iterable = progress(questions, desc='Evaluating') if progress else questions
        results = [
            evaluate_question(q, retriever, top_k_retrieve=args.top_k_retrieve,
                              top_k_rerank=args.top_k_rerank)
            for q in iterable
        ]
    except RetrievalEvalError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2

    summary = summarize(results)
    report = {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'dimension': 'Dimension 2 -- adaptive routing held-out evaluation',
        'split': args.split,
        'benchmark_file': path,
        'processed_dir': args.processed_dir,
        'protocol': {
            'top_k_retrieve': args.top_k_retrieve,
            'top_k_rerank': args.top_k_rerank,
            'evidence_min_top_score': adaptive_config.evidence.min_top_score,
            'reformulation_mode': adaptive_config.reformulation_mode,
            'generation': 'not run -- refusal is fully decided before generation; see module docstring',
        },
        'summary': summary,
        'per_question': [r.to_dict() for r in results],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print()
    print(f"Total questions:         {summary['total_questions']}")
    print(f"  answerable:             {summary['answerable_count']}")
    print(f"  insufficient:           {summary['insufficient_count']}")
    print(f"  ambiguous:              {summary['ambiguous_count']}")
    print(f"  adversarial:            {summary['adversarial_count']}")
    print(f"Routing: direct={summary['routing']['direct']}  "
          f"complex={summary['routing']['complex']} "
          f"(decomposed={summary['routing']['complex_decomposed']})")
    print(f"Evidence: evaluated={summary['evidence']['evaluated']}  "
          f"weak_on_first_attempt={summary['evidence']['weak_evidence_on_first_attempt']}")
    print(f"Reformulations: {summary['reformulations']}")
    print(f"Refusals: {summary['refusals']}")
    print(f"  refusal rate on insufficient:      {summary['refusal_rate_on_insufficient']}")
    print(f"  false-refusal rate on sufficient:  {summary['false_refusal_rate_on_sufficient']}")
    print(f"Errors: {summary['errors']}")
    print(f'\nFull report: {args.output}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
