"""Dimension 3 -- before/after adversarial evaluation.

Runs the adversarial benchmark (data/security/adversarial_benchmark.json)
through the production flow twice:

    baseline  guardrails OFF  == the pipeline exactly as it was before Dimension 3
    guarded   guardrails ON   == what api/rag_api.py now runs by default

and reports Attack Success Rate (ASR) for each. See docs/security.md and the
docstring of evaluation/security_eval.py for the exact definition.

Modes
    offline  No network, no API keys. Retrieval is replaced by the case's own
             clean passage and the LLM by a worst-case "obeys any instruction it
             can still see" simulator. The baseline is therefore 100% ASR BY
             CONSTRUCTION; the guarded number measures how much attack text the
             deterministic layers remove. Prompt hardening is not modelled.
    live     Real AdaptiveRetriever (BM25 + FAISS + Cohere + evidence gate) and
             the real Groq model with the production call parameters. Needs
             COHERE_API_KEY and GROQ_API_KEY. This is the number that means
             something about the deployed system.

Examples
    python scripts/evaluate_security.py --mode offline --ablation
    python scripts/evaluate_security.py --mode live --arms baseline
    python scripts/evaluate_security.py --mode live --arms guarded     # next day is fine: resumes

Live mode appends every finished case to a checkpoint file, so a Groq
daily-token limit or a crash loses nothing: re-run the same command and it
continues. Use --fresh to discard the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from evaluation.security_eval import (
    ABLATION_ARMS,
    EVAL_CANARY,
    SecurityEvalError,
    SimulatedCompliantLLM,
    StaticRetriever,
    compare,
    load_cases,
    render_markdown,
    run_arm,
    summarize,
)

DEFAULT_BENCHMARK = os.path.join('data', 'security', 'adversarial_benchmark.json')
DEFAULT_PROCESSED_DIR = os.path.join('data', 'processed')
DEFAULT_TOP_K_RERANK = 5          # production /query default
DEFAULT_TOP_K_RETRIEVE = 50


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('offline', 'live'), default='offline')
    p.add_argument('--benchmark', default=DEFAULT_BENCHMARK)
    p.add_argument('--output', default=None,
                   help='JSON report path (default data/security/security_eval_<mode>.json)')
    p.add_argument('--arms', default='baseline,guarded',
                   help='comma list: baseline, guarded, only_input_guard, only_context_sanitizer, '
                        'only_prompt_hardening, only_output_guard')
    p.add_argument('--ablation', action='store_true', help='also run each layer on its own')
    p.add_argument('--categories', default=None, help='comma list of attack categories to keep')
    p.add_argument('--limit', type=int, default=None, help='first N cases only (smoke test)')
    p.add_argument('--top-k', type=int, default=DEFAULT_TOP_K_RERANK)
    p.add_argument('--processed-dir', default=DEFAULT_PROCESSED_DIR)
    p.add_argument('--checkpoint', default=None,
                   help='live-mode checkpoint (default data/security/security_eval_live.checkpoint.jsonl)')
    p.add_argument('--fresh', action='store_true', help='delete the checkpoint first')
    return p


def select_cases(doc: Dict, categories: Optional[str], limit: Optional[int]) -> List[Dict]:
    cases = doc['cases']
    if categories:
        keep = {c.strip() for c in categories.split(',') if c.strip()}
        cases = [c for c in cases if c['category'] in keep or c['kind'] == 'benign']
    if limit is not None:
        cases = cases[:limit]
    return cases


def resolve_arms(arms: str, ablation: bool) -> List[str]:
    names = [a.strip() for a in arms.split(',') if a.strip()]
    if ablation:
        names += [a for a in ABLATION_ARMS if a not in names]
    if not names:
        raise SecurityEvalError('no arms selected')
    return names


# ── live wiring (imports are lazy so offline mode needs no heavy dependencies) ─
def build_live(args) -> Dict[str, Callable]:
    from evaluation.retrieval_eval import build_retrievers, load_production_components
    from generation.llm_client import RAGGenerator
    from retrieval.adaptive import AdaptiveConfig, build_adaptive_retriever

    for var in ('COHERE_API_KEY', 'GROQ_API_KEY'):
        if not os.getenv(var):
            raise SecurityEvalError(f'{var} is not set; live mode needs real retrieval and generation.')
    store, bm25, embedder = load_production_components(args.processed_dir)
    hybrid = build_retrievers(store, bm25, embedder, configs=['hybrid_cohere'])['hybrid_cohere']
    # Same construction as api/rag_api.py: production AdaptiveConfig, calibrated
    # evidence threshold untouched.
    retriever = build_adaptive_retriever(hybrid, AdaptiveConfig.from_env(), generator=None)
    generator = RAGGenerator(provider='groq')
    cache: Dict[str, object] = {}     # retrieval does not depend on guardrails: pay for it once

    def retrieve(question: str):
        if question not in cache:
            cache[question] = retriever.retrieve(
                question, top_k_retrieve=DEFAULT_TOP_K_RETRIEVE, top_k_rerank=args.top_k)
        return cache[question]

    def generate(messages: List[Dict]) -> str:
        last = None
        for attempt in range(3):       # same call parameters as api/rag_api.py
            try:
                resp = generator.client.chat.completions.create(
                    model=generator.model, messages=messages, temperature=0.0,
                    max_tokens=1024, timeout=30, reasoning_effort='low')
                return resp.choices[0].message.content.strip()
            except Exception as exc:   # noqa: BLE001
                last = exc
                if 'rate_limit' in str(exc).lower() or '429' in str(exc):
                    raise              # daily token cap: stop and let the checkpoint resume tomorrow
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f'generation failed: {last}')

    return {'retrieve': retrieve, 'generate': generate}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        doc = load_cases(args.benchmark)
        cases = select_cases(doc, args.categories, args.limit)
        arms = resolve_arms(args.arms, args.ablation)
        output = args.output or os.path.join('data', 'security', f'security_eval_{args.mode}.json')

        if args.mode == 'offline':
            retriever_for = StaticRetriever
            llm_for = lambda case: SimulatedCompliantLLM(case)          # noqa: E731
            checkpoint = None
        else:
            live = build_live(args)
            retriever_for = lambda case: live['retrieve']                # noqa: E731
            llm_for = lambda case: live['generate']                      # noqa: E731
            checkpoint = args.checkpoint or os.path.join('data', 'security', 'security_eval_live.checkpoint.jsonl')
            if args.fresh and os.path.exists(checkpoint):
                os.remove(checkpoint)

        print(f"Benchmark: {args.benchmark}  ({sum(c['kind'] == 'attack' for c in cases)} attacks, "
              f"{sum(c['kind'] == 'benign' for c in cases)} benign controls)  mode={args.mode}")
        per_arm: Dict[str, List[Dict]] = {}
        for arm in arms:
            print(f'-- arm: {arm}')
            per_arm[arm] = run_arm(cases, arm, retriever_for=retriever_for, llm_for=llm_for,
                                   checkpoint=checkpoint,
                                   progress=(lambda m: print('  ' + m)) if args.mode == 'live' else None)
    except SecurityEvalError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    except Exception as exc:          # noqa: BLE001  (e.g. rate limit: progress is checkpointed)
        print(f'STOPPED: {exc}\nCompleted cases are saved in the checkpoint; re-run the same command to resume.',
              file=sys.stderr)
        return 3

    summaries = {arm: summarize(rows) for arm, rows in per_arm.items()}
    comparison = compare(summaries['baseline'], summaries['guarded']) \
        if 'baseline' in summaries and 'guarded' in summaries else None

    report = {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'dimension': 'Dimension 3 -- security & adversarial guardrails',
        'mode': args.mode,
        'benchmark_file': args.benchmark,
        'eval_canary': EVAL_CANARY,
        'protocol': {
            'top_k_rerank': args.top_k,
            'arms': arms,
            'asr_definition': 'attacks achieving their goal on the final answer / attack cases completed',
            'offline_caveat': ('simulated worst-case LLM; baseline ASR is 100% by construction; '
                               'prompt hardening not modelled; evidence gate not simulated')
            if args.mode == 'offline' else None,
        },
        'summary': summaries,
        'comparison': comparison,
        'per_case': per_arm,
    }
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')

    print()
    print(render_markdown(args.mode, summaries, comparison))
    if comparison:
        print(f"ASR {comparison['asr_before']:.1%} -> {comparison['asr_after']:.1%} "
              f"(absolute -{comparison['absolute_reduction']:.1%}, relative -{comparison['relative_reduction']:.1%})")
    errs = {a: s['n_errors'] for a, s in summaries.items() if s['n_errors']}
    if errs:
        print(f'WARNING: cases with errors (excluded from ASR denominators): {errs}')
    print(f'Full report: {output}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
