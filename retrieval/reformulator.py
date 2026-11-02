"""Query reformulation for the single retry cycle (Dimension 2).

A reformulator turns a question whose first retrieval produced weak
evidence into *new* queries to retrieve with. Contract:

    reformulate(question, prior_queries, decision) -> List[str]

  * returns only queries that are not (after normalization) already in
    `prior_queries`, so a retry never repeats work already done;
  * may return [] — meaning "nothing distinct to try", in which case the
    caller skips the retry rather than re-running an identical retrieval;
  * never raises for ordinary failures.

Two implementations:

  HeuristicReformulator  deterministic, offline, the default. It strips
                         question scaffolding and filler words to produce
                         keyword-style queries. Honest expectation: this
                         changes the dense embedding and BM25 term weights
                         a little; it will not repair a vocabulary mismatch.
  LLMReformulator        opt-in (ADAPTIVE_REFORMULATION_MODE=llm). One call
                         through the existing Groq client — only ever made
                         on the weak-evidence path, never per query — with
                         automatic fallback to the heuristic one.
"""
import json
import re
from typing import List, Optional, Sequence

from performance.recorder import PURPOSE_REFORMULATION, record_llm_usage
from retrieval.router import RoutingDecision, normalize_query

_STOPWORDS = frozenset('''
a an the and or but if then of in on at to for from by with about into over under
between among across is are was were be been being am do does did done doing have
has had having can could would should will shall may might must what which who whom
whose when where why how whether that this these those there here i me my we our you
your he she it its they them their please tell explain describe give list show find
provide mention state summarize summarise also just very
'''.split())

_TOKEN_RE = re.compile(r"\w+(?:[-'.]\w+)*")


def keyword_query(text: str) -> Optional[str]:
    """Keep content-bearing tokens (in original order/casing); drop
    question scaffolding. Negations ('not', 'no') are deliberately kept
    because they change meaning. Returns None if fewer than two tokens
    survive (too little left to be a useful query)."""
    kept = [t for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]
    return ' '.join(kept) if len(kept) >= 2 else None


class HeuristicReformulator:
    def __init__(self, max_queries: int = 3):
        self.max_queries = max_queries

    def reformulate(self, question: str, prior_queries: Sequence[str],
                    decision: Optional[RoutingDecision] = None) -> List[str]:
        seen = {normalize_query(q) for q in prior_queries}
        sources = [question]
        if decision is not None and decision.decomposed:
            sources += list(decision.subqueries)

        out: List[str] = []
        for source in sources:
            candidate = keyword_query(source)
            if candidate is None:
                continue
            key = normalize_query(candidate)
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
            if len(out) >= self.max_queries:
                break
        return out


_LLM_SYSTEM_PROMPT = (
    'You rewrite search queries for a document retrieval system. The first '
    'search for the question returned weak results. Propose alternative '
    'search queries that use different wording (synonyms, more specific '
    'terms, or the question split into its parts). Do not answer the '
    'question. Respond with ONLY a JSON array of 1 to 3 strings.'
)


class LLMReformulator:
    """Rewrites via the existing Groq chat client (`generator.client`).

    Exactly one attempt: any error, non-JSON output, or an answer with no
    new distinct query falls back to `fallback`. There is no retry loop.
    """

    def __init__(self, client, model: str,
                 fallback: Optional[HeuristicReformulator] = None,
                 max_queries: int = 3, timeout: int = 30):
        self.client = client
        self.model = model
        self.fallback = fallback or HeuristicReformulator(max_queries=max_queries)
        self.max_queries = max_queries
        self.timeout = timeout

    def reformulate(self, question: str, prior_queries: Sequence[str],
                    decision: Optional[RoutingDecision] = None) -> List[str]:
        try:
            queries = self._ask_llm(question, prior_queries)
        except Exception as e:
            print(f'LLM reformulation failed ({e}) — using heuristic reformulation')
            queries = []
        return queries or self.fallback.reformulate(question, prior_queries, decision)

    def _ask_llm(self, question: str, prior_queries: Sequence[str]) -> List[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': _LLM_SYSTEM_PROMPT},
                {'role': 'user', 'content': f'Question: {question}'},
            ],
            temperature=0.0,
            max_tokens=256,
            timeout=self.timeout,
            reasoning_effort='low',
        )
        record_llm_usage(PURPOSE_REFORMULATION, response)   # D5-Lite token accounting
        raw = (response.choices[0].message.content or '').strip()
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.IGNORECASE).strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError('expected a JSON array')

        seen = {normalize_query(q) for q in prior_queries}
        out: List[str] = []
        for item in parsed:
            if not isinstance(item, str):
                continue
            query = ' '.join(item.split())[:300]
            key = normalize_query(query)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(query)
            if len(out) >= self.max_queries:
                break
        return out
