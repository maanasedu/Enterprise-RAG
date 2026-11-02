"""Dimension 1 -- retrieval evaluation.

Scores three retrieval configurations against ``gold_chunk_ids`` from the
benchmark, using the *production* retrieval classes (``FAISSStore``,
``BM25Retriever``, ``HybridRetriever``, ``CohereReranker``) and the production
chunk identity function (``retrieval.identity.chunk_identity_key``). Nothing
in here re-implements ranking, fusion or reranking.

Configurations (all built from ``HybridRetriever``):

* ``dense``          -- dense (FAISS) results only. BM25 is replaced by a stub
                        that returns nothing, so production RRF sees a single
                        list and preserves the dense order. Reranker disabled.
* ``dense_bm25``     -- dense + BM25 fused by production RRF. Reranker disabled.
* ``hybrid_cohere``  -- the exact production pipeline: dense + BM25 + RRF +
                        ``CohereReranker``.

Metrics (per question, then macro-averaged over *scorable* questions):
Recall@5, Recall@10, MRR (= MRR@10), nDCG@5. See docs/retrieval_evaluation.md
for the full protocol, including how unanswerable / insufficient / ambiguous
questions are treated.

This module has NO tunable hyperparameters: retrieval depth, candidate pool
size and RRF ``k`` are the production defaults. Nothing here is fitted on any
split. The held-out ``test`` split can only be run with an explicit
``--confirm-final-test`` flag.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.identity import chunk_identity_key

# --------------------------------------------------------------------------
# Protocol constants (fixed; deliberately not CLI-tunable)
# --------------------------------------------------------------------------
EVAL_TOP_K_RETRIEVE = 50      # production default candidate pool (HybridRetriever.retrieve)
EVAL_DEPTH = 10               # number of ranked chunks scored per question
RECALL_KS = (5, 10)
NDCG_K = 5
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 2000

METRIC_KEYS = ('recall@5', 'recall@10', 'mrr', 'ndcg@5')
SUPPLEMENTARY_KEYS = ('all_gold@5', 'all_gold@10')
METRIC_HEADERS = {
    'recall@5': 'Recall@5',
    'recall@10': 'Recall@10',
    'mrr': 'MRR',
    'ndcg@5': 'nDCG@5',
}

CONFIG_LABELS = {
    'dense': 'Dense Only',
    'dense_bm25': 'Dense + BM25',
    'hybrid_cohere': 'Hybrid + Cohere Reranker',
}
CONFIG_ORDER = tuple(CONFIG_LABELS)
SPLITS = ('calibration', 'test')

DEFAULT_BENCHMARK_DIR = os.path.join('data', 'benchmark')
DEFAULT_PROCESSED_DIR = os.path.join('data', 'processed')

_REQUIRED_FIELDS = (
    'id', 'category', 'question', 'answerable', 'evidence_status',
    'gold_chunk_ids', 'gold_document_ids',
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class RetrievalEvalError(Exception):
    """Base class for all expected, user-facing evaluation errors."""


class HeldOutSplitError(RetrievalEvalError):
    """Raised when the held-out test split is requested without confirmation."""


class MissingCredentialsError(RetrievalEvalError):
    """Raised when a requested configuration needs credentials that are absent."""


class BenchmarkDataError(RetrievalEvalError):
    """Raised for malformed/inconsistent benchmark data or a corpus mismatch."""


class IndexNotAvailableError(RetrievalEvalError):
    """Raised when the production FAISS/BM25 artifacts are missing or unusable."""


class RerankerFallbackError(RetrievalEvalError):
    """Raised when the Cohere config silently degraded to position fallback."""


# --------------------------------------------------------------------------
# Metrics (binary relevance; a chunk is relevant iff its identity key is gold)
# --------------------------------------------------------------------------
def dedupe_ranking(keys: Iterable[str]) -> List[str]:
    """Drop repeated identity keys, keeping the first (best) occurrence."""
    seen = set()
    out: List[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _require_gold(gold: Iterable[str]) -> set:
    gold_set = set(gold)
    if not gold_set:
        raise ValueError(
            'Retrieval metrics are undefined for a question with no gold chunks; '
            'exclude it before scoring.'
        )
    return gold_set


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of the gold chunks that appear in the top-k.

    For a question with two gold chunks (multi-document), finding one of them
    scores 0.5 and finding both scores 1.0.
    """
    gold_set = _require_gold(gold)
    top = set(dedupe_ranking(ranked)[:k])
    return len(gold_set & top) / len(gold_set)


def all_gold_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 iff *every* gold chunk is in the top-k (strict multi-doc coverage)."""
    gold_set = _require_gold(gold)
    top = set(dedupe_ranking(ranked)[:k])
    return 1.0 if gold_set <= top else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str],
                    depth: int = EVAL_DEPTH) -> float:
    """1 / rank of the first gold chunk within the first ``depth`` results."""
    gold_set = _require_gold(gold)
    for rank, key in enumerate(dedupe_ranking(ranked)[:depth], start=1):
        if key in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int = NDCG_K) -> float:
    """nDCG@k with binary gains and log2(rank + 1) discount.

    The ideal ranking places ``min(|gold|, k)`` relevant chunks first, so a
    question with two gold chunks is only scored 1.0 when both are on top.
    """
    gold_set = _require_gold(gold)
    top = dedupe_ranking(ranked)[:k]
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, key in enumerate(top, start=1) if key in gold_set)
    ideal = sum(1.0 / math.log2(rank + 1)
                for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal


def score_question(ranked: Sequence[str], gold: Iterable[str],
                   depth: int = EVAL_DEPTH) -> Dict[str, float]:
    gold_set = _require_gold(gold)
    ranked = dedupe_ranking(ranked)
    return {
        'recall@5': recall_at_k(ranked, gold_set, 5),
        'recall@10': recall_at_k(ranked, gold_set, 10),
        'mrr': reciprocal_rank(ranked, gold_set, depth),
        'ndcg@5': ndcg_at_k(ranked, gold_set, NDCG_K),
        'all_gold@5': all_gold_at_k(ranked, gold_set, 5),
        'all_gold@10': all_gold_at_k(ranked, gold_set, 10),
    }


def bootstrap_ci(values: Sequence[float], n_resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean (deterministic given ``seed``)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError('Cannot bootstrap an empty sample.')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def aggregate(per_question: Sequence[Dict]) -> Dict:
    """Macro-average scores (and bootstrap CIs) over per-question results."""
    if not per_question:
        raise ValueError('No scored questions to aggregate.')
    metrics, ci95, supplementary = {}, {}, {}
    for key in METRIC_KEYS:
        vals = [q[key] for q in per_question]
        metrics[key] = float(np.mean(vals))
        ci95[key] = list(bootstrap_ci(vals))
    for key in SUPPLEMENTARY_KEYS:
        supplementary[key] = float(np.mean([q[key] for q in per_question]))
    return {'metrics': metrics, 'ci95': ci95, 'supplementary': supplementary,
            'n': len(per_question)}


# --------------------------------------------------------------------------
# Benchmark loading
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    category: str
    question: str
    answerable: bool
    evidence_status: str
    gold_chunk_ids: Tuple[str, ...]
    gold_document_ids: Tuple[str, ...]

    @property
    def scorable(self) -> bool:
        """True iff chunk-level retrieval ground truth exists for this question."""
        return len(self.gold_chunk_ids) > 0


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def split_path(benchmark_dir: str, split: str) -> str:
    if split not in SPLITS:
        raise BenchmarkDataError(f'Unknown split {split!r}; expected one of {SPLITS}.')
    return os.path.join(benchmark_dir, f'{split}.json')


def load_questions(path: str) -> List[BenchmarkQuestion]:
    """Load a benchmark split and enforce the scoring invariants.

    * ``evidence_status == 'sufficient'`` <=> non-empty ``gold_chunk_ids``.
      Any other combination is a data error (we refuse rather than guess).
    * question ids must be unique.
    """
    if not os.path.isfile(path):
        raise BenchmarkDataError(
            f'Benchmark file not found: {path}. Place benchmark.json / '
            f'calibration.json / test.json in that directory or pass --benchmark-dir.'
        )
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise BenchmarkDataError(f'{path}: expected a JSON list of questions.')

    questions: List[BenchmarkQuestion] = []
    seen_ids = set()
    for i, item in enumerate(raw):
        missing = [k for k in _REQUIRED_FIELDS if k not in item]
        if missing:
            raise BenchmarkDataError(f'{path}[{i}]: missing fields {missing}.')
        if item['id'] in seen_ids:
            raise BenchmarkDataError(f'{path}: duplicate question id {item["id"]}.')
        seen_ids.add(item['id'])
        has_gold = len(item['gold_chunk_ids']) > 0
        sufficient = item['evidence_status'] == 'sufficient'
        if has_gold != sufficient:
            raise BenchmarkDataError(
                f'{path}: {item["id"]} has evidence_status={item["evidence_status"]!r} '
                f'but {len(item["gold_chunk_ids"])} gold_chunk_ids; refusing to score.'
            )
        questions.append(BenchmarkQuestion(
            id=item['id'],
            category=item['category'],
            question=item['question'],
            answerable=bool(item['answerable']),
            evidence_status=item['evidence_status'],
            gold_chunk_ids=tuple(item['gold_chunk_ids']),
            gold_document_ids=tuple(item['gold_document_ids']),
        ))
    return questions


def assert_gold_in_corpus(questions: Sequence[BenchmarkQuestion],
                          metadata_records: Sequence[Dict]) -> None:
    """Fail loudly if any gold chunk id is absent from the loaded corpus.

    Keys are computed with the production ``chunk_identity_key``. This catches
    a re-built index that no longer matches the benchmark.
    """
    corpus_keys = {chunk_identity_key(rec) for rec in metadata_records}
    missing = sorted({g for q in questions for g in q.gold_chunk_ids} - corpus_keys)
    if missing:
        raise BenchmarkDataError(
            f'{len(missing)} gold chunk id(s) are not in the loaded corpus '
            f'(first: {missing[0]}). The index no longer matches the benchmark.'
        )


def guard_split(split: str, confirm_final_test: bool) -> None:
    """The held-out split needs an explicit acknowledgement to be evaluated."""
    if split not in SPLITS:
        raise BenchmarkDataError(f'Unknown split {split!r}; expected one of {SPLITS}.')
    if split == 'test' and not confirm_final_test:
        raise HeldOutSplitError(
            'test.json is strictly held out and must not be used for tuning. '
            'Re-run with --confirm-final-test only for the final, one-shot evaluation.'
        )


def summarize_exclusions(questions: Sequence[BenchmarkQuestion]) -> Dict:
    scored = [q for q in questions if q.scorable]
    excluded = [q for q in questions if not q.scorable]

    def _count(items, attr):
        out: Dict[str, int] = {}
        for q in items:
            key = getattr(q, attr)
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    return {
        'n_questions': len(questions),
        'n_scored': len(scored),
        'n_excluded': len(excluded),
        'scored_by_category': _count(scored, 'category'),
        'excluded_by_category': _count(excluded, 'category'),
        'excluded_by_evidence_status': _count(excluded, 'evidence_status'),
    }


# --------------------------------------------------------------------------
# Retrieval configurations (production classes only)
# --------------------------------------------------------------------------
class _NullBM25:
    """BM25 stand-in that returns nothing, used to derive the dense-only config.

    ``HybridRetriever`` then fuses a single (dense) list with production RRF,
    which preserves the dense ordering -- no separate retrieval code path.
    """

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        return []


def build_retrievers(dense_store, bm25, embedder,
                     configs: Sequence[str] = CONFIG_ORDER,
                     reranker=None) -> Dict[str, HybridRetriever]:
    """Build the requested configurations from the production HybridRetriever.

    ``reranker`` is an optional injection point (used by tests to supply a
    mocked Cohere reranker). When omitted, ``hybrid_cohere`` uses the
    production constructor path (``use_reranker=True`` -> ``CohereReranker()``).
    """
    unknown = [c for c in configs if c not in CONFIG_LABELS]
    if unknown:
        raise ValueError(f'Unknown configuration(s): {unknown}')
    retrievers: Dict[str, HybridRetriever] = {}
    for name in configs:
        if name == 'dense':
            retrievers[name] = HybridRetriever(
                dense_store, _NullBM25(), embedder, use_reranker=False)
        elif name == 'dense_bm25':
            retrievers[name] = HybridRetriever(
                dense_store, bm25, embedder, use_reranker=False)
        else:  # hybrid_cohere
            hybrid = HybridRetriever(
                dense_store, bm25, embedder, use_reranker=(reranker is None))
            if reranker is not None:
                hybrid.reranker = reranker
            retrievers[name] = hybrid
    return retrievers


def retrieve_ranking(retriever: HybridRetriever, question: str, config: str, *,
                     depth: int = EVAL_DEPTH,
                     top_k_retrieve: int = EVAL_TOP_K_RETRIEVE,
                     allow_rerank_fallback: bool = False) -> Tuple[List[str], int]:
    """Run production retrieval and return (ranked identity keys, #unreranked).

    ``#unreranked`` is only meaningful for ``hybrid_cohere``: it counts results
    that the production reranker marked ``reranked=False`` (no key / API
    error). Unless explicitly allowed, that raises, because a silent fallback
    would report Dense+BM25 numbers under the Cohere label.
    """
    results = retriever.retrieve(question, top_k_retrieve=top_k_retrieve,
                                 top_k_rerank=depth)
    n_unreranked = 0
    if config == 'hybrid_cohere':
        n_unreranked = sum(1 for r in results if not r.get('reranked'))
        if n_unreranked and not allow_rerank_fallback:
            raise RerankerFallbackError(
                'Cohere reranking fell back to position order (missing/invalid '
                'COHERE_API_KEY or an API error). Refusing to report these '
                'results as "Hybrid + Cohere Reranker".'
            )
    ranking = dedupe_ranking(chunk_identity_key(r) for r in results)
    return ranking, n_unreranked


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_config(config: str, retriever: HybridRetriever,
                    questions: Sequence[BenchmarkQuestion], *,
                    depth: int = EVAL_DEPTH,
                    top_k_retrieve: int = EVAL_TOP_K_RETRIEVE,
                    allow_rerank_fallback: bool = False,
                    progress: Optional[Callable] = None) -> Dict:
    """Evaluate one configuration over the scorable questions only.

    Unscorable questions (no gold chunks) are never sent to the retriever.
    """
    scorable = [q for q in questions if q.scorable]
    if not scorable:
        raise BenchmarkDataError('No scorable questions (all have empty gold_chunk_ids).')
    iterator = progress(scorable, desc=CONFIG_LABELS[config]) if progress else scorable

    per_question: List[Dict] = []
    fallback_questions = 0
    for q in iterator:
        ranking, n_unreranked = retrieve_ranking(
            retriever, q.question, config, depth=depth,
            top_k_retrieve=top_k_retrieve, allow_rerank_fallback=allow_rerank_fallback)
        if n_unreranked:
            fallback_questions += 1
        scores = score_question(ranking, q.gold_chunk_ids, depth)
        per_question.append({
            'id': q.id,
            'category': q.category,
            'n_gold': len(set(q.gold_chunk_ids)),
            'gold_chunk_ids': list(q.gold_chunk_ids),
            'retrieved_chunk_ids': ranking,
            **scores,
        })

    overall = aggregate(per_question)
    by_category = {}
    for category in sorted({p['category'] for p in per_question}):
        subset = [p for p in per_question if p['category'] == category]
        by_category[category] = {
            'n': len(subset),
            'metrics': {k: float(np.mean([p[k] for p in subset])) for k in METRIC_KEYS},
        }
    return {
        'label': CONFIG_LABELS[config],
        'n_scored': overall['n'],
        'metrics': overall['metrics'],
        'ci95': overall['ci95'],
        'supplementary': overall['supplementary'],
        'by_category': by_category,
        'rerank_fallback_questions': fallback_questions,
        'reranker_model': getattr(getattr(retriever, 'reranker', None), 'model', None)
        if config == 'hybrid_cohere' else None,
        'per_question': per_question,
    }


def _rrf_k_default() -> int:
    return inspect.signature(
        HybridRetriever._reciprocal_rank_fusion).parameters['k'].default


def protocol_description(depth: int, top_k_retrieve: int) -> Dict:
    return {
        'top_k_retrieve': top_k_retrieve,
        'depth_scored': depth,
        'rrf_k': _rrf_k_default(),
        'recall_ks': list(RECALL_KS),
        'ndcg_k': NDCG_K,
        'mrr': f'MRR@{depth} (0 when no gold chunk is within the top {depth})',
        'relevance': 'binary; a retrieved chunk is relevant iff chunk_identity_key(chunk) is in gold_chunk_ids',
        'aggregation': 'macro-average over scorable questions',
        'excluded': 'questions with empty gold_chunk_ids (unanswerable, insufficient, ambiguous, adversarial-insufficient) are not scored',
        'hyperparameters_tuned': False,
        'bootstrap': {'seed': BOOTSTRAP_SEED, 'resamples': BOOTSTRAP_RESAMPLES,
                      'interval': 'percentile 95%'},
    }


def run_evaluation(questions: Sequence[BenchmarkQuestion],
                   retrievers: Dict[str, HybridRetriever], *,
                   split: str, benchmark_path: Optional[str] = None,
                   depth: int = EVAL_DEPTH,
                   top_k_retrieve: int = EVAL_TOP_K_RETRIEVE,
                   allow_rerank_fallback: bool = False,
                   progress: Optional[Callable] = None) -> Dict:
    """Evaluate every retriever in ``retrievers`` and return a full report."""
    report = {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'split': split,
        'benchmark_file': benchmark_path,
        'benchmark_sha256': file_sha256(benchmark_path) if benchmark_path
        and os.path.isfile(benchmark_path) else None,
        'protocol': protocol_description(depth, top_k_retrieve),
        'counts': summarize_exclusions(questions),
        'configs': {},
    }
    for name in CONFIG_ORDER:
        if name in retrievers:
            report['configs'][name] = evaluate_config(
                name, retrievers[name], questions, depth=depth,
                top_k_retrieve=top_k_retrieve,
                allow_rerank_fallback=allow_rerank_fallback, progress=progress)
    return report


def format_results_table(report: Dict) -> str:
    counts = report['counts']
    lines = [
        f"Split: {report['split']} | scored: {counts['n_scored']} | "
        f"excluded (no gold chunks): {counts['n_excluded']} | total: {counts['n_questions']}",
        '',
        '| Configuration | ' + ' | '.join(METRIC_HEADERS[k] for k in METRIC_KEYS) + ' |',
        '|---|' + '---:|' * len(METRIC_KEYS),
    ]
    for name in CONFIG_ORDER:
        cfg = report['configs'].get(name)
        if not cfg:
            continue
        cells = ' | '.join(f"{cfg['metrics'][k]:.3f}" for k in METRIC_KEYS)
        lines.append(f"| {cfg['label']} | {cells} |")
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Production component loading (mirrors api/rag_api.py + scripts/run_evaluation.py)
# --------------------------------------------------------------------------
def _reject_lfs_pointer(path: str) -> None:
    if not os.path.isfile(path):
        raise IndexNotAvailableError(f'Required artifact not found: {path}')
    if os.path.getsize(path) < 1024:
        with open(path, 'rb') as f:
            head = f.read(64)
        if head.startswith(b'version https://git-lfs'):
            raise IndexNotAvailableError(
                f'{path} is a Git LFS pointer, not the real file. '
                f'Run `git lfs pull` (or rebuild the index with scripts/ingest.py).'
            )


def load_production_components(processed_dir: str = DEFAULT_PROCESSED_DIR):
    """Load the same FAISS index, metadata, BM25 index and embedder as production."""
    index_path = os.path.join(processed_dir, 'faiss_index.bin')
    meta_path = os.path.join(processed_dir, 'faiss_metadata.json')
    bm25_path = os.path.join(processed_dir, 'bm25_index.pkl')
    # Validate artifacts first so a bad checkout fails fast, before the
    # (slow) embedding-model import.
    for p in (index_path, bm25_path):
        _reject_lfs_pointer(p)
    if not os.path.isfile(meta_path):
        raise IndexNotAvailableError(f'Required artifact not found: {meta_path}')

    import pickle
    import faiss as faiss_lib
    from ingestion.embedder import CachedEmbedder
    from retrieval.vector_store import FAISSStore

    store = FAISSStore(dim=1024)
    store.index = faiss_lib.read_index(index_path)
    with open(meta_path, encoding='utf-8') as f:
        store.metadata = json.load(f)
    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)

    n_index, n_meta, n_bm25 = store.index.ntotal, len(store.metadata), len(bm25.corpus)
    if not (n_index == n_meta == n_bm25):
        raise IndexNotAvailableError(
            f'Index/metadata/BM25 size mismatch: faiss={n_index}, '
            f'metadata={n_meta}, bm25={n_bm25}.'
        )
    return store, bm25, CachedEmbedder()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Dimension 1 retrieval evaluation (production retrieval code).')
    p.add_argument('--split', choices=SPLITS, default='calibration',
                   help='Benchmark split to evaluate (default: calibration).')
    p.add_argument('--confirm-final-test', action='store_true',
                   help='Required to evaluate the held-out test split.')
    p.add_argument('--configs', nargs='+', choices=CONFIG_ORDER, default=list(CONFIG_ORDER),
                   help='Configurations to run (default: all three).')
    p.add_argument('--benchmark-dir', default=DEFAULT_BENCHMARK_DIR)
    p.add_argument('--processed-dir', default=DEFAULT_PROCESSED_DIR)
    p.add_argument('--output-dir', default=DEFAULT_PROCESSED_DIR)
    p.add_argument('--limit', type=int, default=None,
                   help='Smoke-test only: evaluate the first N scorable questions.')
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
        guard_split(args.split, args.confirm_final_test)
        if 'hybrid_cohere' in args.configs and not os.getenv('COHERE_API_KEY', ''):
            raise MissingCredentialsError(
                'COHERE_API_KEY is not set; the "Hybrid + Cohere Reranker" '
                'configuration needs it. Set it, or run with '
                '--configs dense dense_bm25.')

        path = split_path(args.benchmark_dir, args.split)
        questions = load_questions(path)
        if args.limit is not None:
            scorable = [q for q in questions if q.scorable][:args.limit]
            questions = scorable
        print(f'Loaded {len(questions)} question(s) from {path}')

        store, bm25, embedder = load_production_components(args.processed_dir)
        assert_gold_in_corpus(questions, store.metadata)

        retrievers = build_retrievers(store, bm25, embedder, args.configs)
        report = run_evaluation(questions, retrievers, split=args.split,
                                benchmark_path=path, progress=_progress_factory())
    except RetrievalEvalError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f'_limit{args.limit}' if args.limit is not None else ''
    out_path = os.path.join(args.output_dir, f'retrieval_eval_{args.split}{suffix}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print()
    print(format_results_table(report))
    print(f'\nFull report (per-question results, CIs, per-category): {out_path}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
