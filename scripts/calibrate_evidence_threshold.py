"""Dimension 2 -- calibrate retrieval/evidence.py's evidence threshold.

Runs the 48-question CALIBRATION split (data/evaluation/calibration.json)
through the *existing, unmodified* production pipeline -- QueryRouter ->
HybridRetriever -> CohereReranker -> EvidenceEvaluator, exactly as
retrieval/adaptive.py::AdaptiveRetriever wires them for /query -- to collect
one top reranker score per question, then selects a threshold with
evaluation/threshold_calibration.py's documented rule.

Nothing here re-implements routing, retrieval, reranking or evidence
scoring: this script only supplies an EvidenceConfig that never rejects (so
every question resolves in exactly one retrieve+rerank round, no retry) and
records what the real EvidenceEvaluator computed.

    python scripts/calibrate_evidence_threshold.py

Requires:
  * COHERE_API_KEY (real reranker scores; a mocked/fallback score carries no
    calibration signal -- see MissingCredentialsError below)
  * The real FAISS/BM25 corpus in data/processed/ (not Git LFS pointers --
    run `git lfs pull` first if needed; see IndexNotAvailableError below)

Does NOT touch data/evaluation/test.json (the held-out 202-question set) or
run generation. Does NOT modify retrieval/evidence.py's default threshold --
it only reports what the calibration procedure supports.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Sequence

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
from evaluation.threshold_calibration import (
    CalibrationRecord,
    select_threshold,
)
from retrieval.adaptive import AdaptiveRetriever
from retrieval.evidence import DEFAULT_MIN_TOP_SCORE, EvidenceConfig, EvidenceEvaluator
from retrieval.router import QueryRouter

DEFAULT_BENCHMARK_DIR = os.path.join('data', 'evaluation')
DEFAULT_PROCESSED_DIR = os.path.join('data', 'processed')
DEFAULT_OUTPUT = os.path.join('data', 'evaluation', 'evidence_threshold_calibration.json')

# Production /query default (QueryRequest.top_k in api/rag_api.py).
DEFAULT_TOP_K_RERANK = 5
DEFAULT_TOP_K_RETRIEVE = 50

# A threshold well below any real Cohere score, used ONLY to force the
# EvidenceEvaluator to always accept on the first attempt, so exactly one
# retrieve+rerank round happens per question and its raw top_score can be
# read off (retrieval/adaptive.py's retry path is never triggered).
_NEVER_REFUSE_THRESHOLD = -1_000_000.0


class RerankerFallbackError(RetrievalEvalError):
    """Raised when Cohere silently fell back for one or more questions.

    Mirrors evaluation/retrieval_eval.py's identically-named guard: a
    fallback score is synthetic (derived from candidate rank order, not
    relevance -- see retrieval/reranker.py), so a calibration run that
    includes one is not measuring what it claims to measure.
    """


@dataclass(frozen=True)
class CollectedScore:
    record: CalibrationRecord
    scores_considered: Sequence[float]
    n_candidates: int


def collect_scores(questions, hybrid, *, evidence_top_k: int,
                   top_k_rerank: int, top_k_retrieve: int,
                   progress=None) -> List[CollectedScore]:
    """Run each question through the real routing/retrieval/rerank/evidence
    pipeline (AdaptiveRetriever, unmodified) and record its round-1
    EvidenceVerdict. One Cohere rerank call per question (no retries: the
    evaluator's threshold is set to always accept)."""
    evaluator = EvidenceEvaluator(EvidenceConfig(
        top_k=evidence_top_k,
        min_top_score=_NEVER_REFUSE_THRESHOLD,
        fail_open_when_unscored=True,   # never refuse here; we just want the score
    ))
    retriever = AdaptiveRetriever(hybrid, router=QueryRouter(), evaluator=evaluator)

    iterable = progress(questions, desc='Calibrating') if progress else questions
    out: List[CollectedScore] = []
    for q in iterable:
        result = retriever.retrieve(q.question, top_k_retrieve=top_k_retrieve,
                                    top_k_rerank=top_k_rerank)
        assert len(result.attempts) == 1, (
            'evidence evaluator unexpectedly retried during score collection '
            f'for {q.id!r} -- _NEVER_REFUSE_THRESHOLD is not low enough'
        )
        verdict = result.verdict
        record = CalibrationRecord(
            question_id=q.id,
            evidence_status=q.evidence_status,
            top_score=verdict.top_score,
            evaluated=verdict.evaluated,
            category=q.category,
            routing_path=result.decision.path,
        )
        out.append(CollectedScore(record, verdict.scores_considered, len(result.chunks)))
    return out


def build_report(questions, collected: Sequence[CollectedScore], *,
                 evidence_top_k: int, top_k_rerank: int, top_k_retrieve: int,
                 benchmark_path: str, processed_dir: str) -> dict:
    records = [c.record for c in collected]
    calibration = select_threshold(records, placeholder_threshold=DEFAULT_MIN_TOP_SCORE)

    n_unreranked_fallback = sum(1 for c in collected if not c.record.evaluated)

    return {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'dimension': 'Dimension 2 -- adaptive routing / evidence threshold calibration',
        'split': 'calibration',
        'benchmark_file': benchmark_path,
        'processed_dir': processed_dir,
        'n_questions': len(questions),
        'protocol': {
            'top_k_retrieve': top_k_retrieve,
            'top_k_rerank': top_k_rerank,
            'evidence_top_k': evidence_top_k,
            'reranker_model': 'rerank-english-v3.0',
            'retries': 'disabled for collection (evaluator forced to always accept on round 1)',
            'routing': 'production QueryRouter (LOW=direct, HIGH=decompose+merge+rerank once)',
            'note': (
                'Cohere reranker scores are ranking signals, not probabilities; '
                'thresholds below are compared against raw scores, never rescaled.'
            ),
        },
        'rerank_fallback_questions': n_unreranked_fallback,
        'per_question': [
            {
                'id': c.record.question_id,
                'category': c.record.category,
                'evidence_status': c.record.evidence_status,
                'routing_path': c.record.routing_path,
                'evaluated': c.record.evaluated,
                'top_score': c.record.top_score,
                'scores_considered': list(c.scores_considered),
                'n_candidates_returned': c.n_candidates,
            }
            for c in collected
        ],
        'calibration': calibration.to_dict(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Dimension 2 -- calibrate the evidence threshold on the '
                    'calibration split (production retrieval/evidence code).')
    p.add_argument('--benchmark-dir', default=DEFAULT_BENCHMARK_DIR)
    p.add_argument('--processed-dir', default=DEFAULT_PROCESSED_DIR)
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--top-k-rerank', type=int, default=DEFAULT_TOP_K_RERANK,
                   help='Matches QueryRequest.top_k default in api/rag_api.py.')
    p.add_argument('--top-k-retrieve', type=int, default=DEFAULT_TOP_K_RETRIEVE)
    p.add_argument('--evidence-top-k', type=int, default=EvidenceConfig().top_k)
    p.add_argument('--limit', type=int, default=None,
                   help='Smoke-test only: calibrate on the first N questions.')
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
                'COHERE_API_KEY is not set. Evidence-threshold calibration needs '
                'real Cohere reranker scores -- a fallback score carries no '
                'calibration signal (see retrieval/reranker.py). Set it and re-run.'
            )

        path = split_path(args.benchmark_dir, 'calibration')
        questions = load_questions(path)
        if args.limit is not None:
            questions = questions[:args.limit]
        print(f'Loaded {len(questions)} question(s) from {path}')

        store, bm25, embedder = load_production_components(args.processed_dir)
        assert_gold_in_corpus(questions, store.metadata)

        hybrid = build_retrievers(store, bm25, embedder, configs=['hybrid_cohere'])['hybrid_cohere']

        collected = collect_scores(
            questions, hybrid,
            evidence_top_k=args.evidence_top_k,
            top_k_rerank=args.top_k_rerank,
            top_k_retrieve=args.top_k_retrieve,
            progress=_progress_factory(),
        )

        n_fallback = sum(1 for c in collected if not c.record.evaluated)
        if n_fallback:
            raise RerankerFallbackError(
                f'Cohere reranking fell back to position order for {n_fallback} '
                f'question(s) (missing/invalid COHERE_API_KEY or an API error '
                f'partway through). Refusing to calibrate on partially synthetic '
                f'scores -- fix the API issue and re-run.'
            )

        report = build_report(
            questions, collected,
            evidence_top_k=args.evidence_top_k,
            top_k_rerank=args.top_k_rerank,
            top_k_retrieve=args.top_k_retrieve,
            benchmark_path=path,
            processed_dir=args.processed_dir,
        )
    except RetrievalEvalError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    cal = report['calibration']
    print()
    print(f"Selected threshold: {cal['selected_threshold']}")
    print(f"  (placeholder was: {cal['placeholder_threshold']}, "
          f"replace: {cal['replace_placeholder']})")
    print(f"  sufficient:   {cal['counts']['n_sufficient']} "
          f"(refused: {cal['selected_point']['refused_sufficient']})")
    print(f"  insufficient: {cal['counts']['n_insufficient']} "
          f"(rejected: {cal['selected_point']['refused_insufficient']})")
    print(f"  ambiguous (diagnostic only): {cal['counts']['n_ambiguous_excluded_from_objective']}")
    print(f'\nFull report: {args.output}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
