"""Lightweight, deterministic query router (Dimension 2 — adaptive routing).

Decides how much retrieval work a question needs, using only cheap lexical
signals — no LLM call, no network, no model:

  LOW  -> one direct hybrid retrieval
  HIGH -> decompose into sub-queries, retrieve for each, merge, rerank once

A query is HIGH when at least one signal fires:

  multiple_questions     two or more separate questions ("A? B?")
  enumerated_parts       "(1) ... (2) ...", "1. ... 2. ...", "first ... second"
  multi_part_conjunction "... and how does ...", "...; why ..."
  comparison             compare / vs / difference between / X than Y
  multi_document         "across the documents", "in both reports", ...
  long_query             at least `long_query_words` words

These are heuristics for "a single retrieval query is unlikely to retrieve
all the evidence". They are intentionally simple and inspectable; whether
they actually help is an empirical question for the Dimension 1 benchmark.
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class Complexity(str, Enum):
    LOW = 'low'
    HIGH = 'high'


@dataclass(frozen=True)
class RouterConfig:
    # Word count at/above which a query is considered complex on length alone.
    long_query_words: int = 35
    # Upper bound on sub-queries (and therefore on retrieval calls) per cycle.
    max_subqueries: int = 4
    # Fragments with fewer words are not treated as standalone sub-questions.
    min_subquery_words: int = 2

    def __post_init__(self):
        if self.long_query_words < 1:
            raise ValueError('long_query_words must be >= 1')
        if self.max_subqueries < 1:
            raise ValueError('max_subqueries must be >= 1')
        if self.min_subquery_words < 1:
            raise ValueError('min_subquery_words must be >= 1')


@dataclass(frozen=True)
class RoutingDecision:
    complexity: Complexity
    signals: Tuple[str, ...]
    # Queries to retrieve with. Always non-empty. A single element means
    # "retrieve directly with this one query"; more than one means decompose.
    subqueries: Tuple[str, ...]

    @property
    def is_high(self) -> bool:
        return self.complexity == Complexity.HIGH

    @property
    def decomposed(self) -> bool:
        return len(self.subqueries) > 1

    @property
    def path(self) -> str:
        return 'decomposed' if self.decomposed else 'direct'

    def to_dict(self) -> dict:
        return {
            'complexity': self.complexity.value,
            'path': self.path,
            'signals': list(self.signals),
            'subqueries': list(self.subqueries),
        }


def normalize_query(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — used to detect
    that two query strings are effectively the same query."""
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.lower()).split())


_INTERROGATIVE = r'(?:what|how|why|when|where|who|whom|whose|which|whether)'
_DOC_NOUN = (
    r'(?:documents?|files?|reports?|contracts?|polic(?:y|ies)|sources?|filings?|'
    r'papers?|manuals?|agreements?|articles?)'
)

# "... and how ..." / "... also what ..." / "...; why ..."
_CONJUNCTION_SPLIT_RE = re.compile(
    rf'\s*;\s*(?={_INTERROGATIVE}\b)'
    rf'|\s+(?:and|also|plus|then)\s+(?:also\s+)?(?={_INTERROGATIVE}\b)',
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r'\b(?:compare[sd]?|comparing|comparison|contrast(?:s|ing)?|versus|'
    r'differences?\s+between|differs?\s+from|differ\s+between|similarities|'
    r'(?:better|worse|faster|slower|cheaper|higher|lower|larger|smaller)\s+than)\b'
    r'|\bvs\b\.?',
    re.IGNORECASE,
)

_ENUM_MARKER_RE = re.compile(
    r'(?:(?<=\s)|^)(?:\(\s*(?:\d{1,2}|[a-zA-Z])\s*\)|\d{1,2}[.)])\s+(?=\S)'
)
_ORDINAL_RE = re.compile(r'\bfirst(?:ly)?\b.*\bsecond(?:ly)?\b', re.IGNORECASE | re.DOTALL)

_MULTI_DOC_RE = re.compile(
    rf'\b(?:across|among)\s+(?:(?:the|all|multiple|several|different|various)\s+)*{_DOC_NOUN}\b'
    rf'|\b(?:in|from)\s+(?:both|all|each|every)\s+(?:the\s+)?{_DOC_NOUN}\b'
    rf'|\b(?:multiple|several|various|different)\s+{_DOC_NOUN}\b'
    rf'|\beach\s+(?:of\s+the\s+)?{_DOC_NOUN}\b',
    re.IGNORECASE,
)

# Ordered most- to least-specific; the first pattern that matches wins.
_COMPARISON_ENTITY_PATTERNS = [
    re.compile(r'\bdifferences?\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)[\s?.!]*$', re.IGNORECASE),
    re.compile(
        r'\bcompar(?:e|es|ed|ing|ison)\s+(?:of\s+|between\s+)?(?P<a>.+?)\s+'
        r'(?:and|with|to|against|versus|vs\.?)\s+(?P<b>.+?)[\s?.!]*$',
        re.IGNORECASE,
    ),
    re.compile(
        r'\bhow\s+(?:does|do|did)\s+(?P<a>.+?)\s+'
        r'(?:differ\s+from|compare\s+(?:to|with))\s+(?P<b>.+?)[\s?.!]*$',
        re.IGNORECASE,
    ),
    re.compile(r'(?P<a>.+?)\s+(?:versus|vs\.?)\s+(?P<b>.+?)[\s?.!]*$', re.IGNORECASE),
]


_TRAILING_SCOPE_RE = re.compile(
    r'^(?P<entity>.+?)\s+(?P<scope>(?:on|for|in|regarding|about|when it comes to|'
    r'in terms of|with respect to)\s+.+)$',
    re.IGNORECASE,
)


class QueryRouter:
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()

    # ── public API ──────────────────────────────────────────────────────────
    def route(self, query: str) -> RoutingDecision:
        text = ' '.join((query or '').split())
        if not text:
            return RoutingDecision(Complexity.LOW, (), (query or '',))

        signals = self._detect_signals(text)
        if not signals:
            return RoutingDecision(Complexity.LOW, (), (text,))

        subqueries = self._decompose(text, signals)
        return RoutingDecision(Complexity.HIGH, tuple(signals), tuple(subqueries))

    # ── signal detection ────────────────────────────────────────────────────
    def _detect_signals(self, text: str) -> List[str]:
        signals: List[str] = []
        if len(self._split_questions(text)) >= 2:
            signals.append('multiple_questions')
        if len(_ENUM_MARKER_RE.findall(text)) >= 2 or _ORDINAL_RE.search(text):
            signals.append('enumerated_parts')
        if _CONJUNCTION_SPLIT_RE.search(text):
            signals.append('multi_part_conjunction')
        if _COMPARISON_RE.search(text):
            signals.append('comparison')
        if _MULTI_DOC_RE.search(text):
            signals.append('multi_document')
        if len(text.split()) >= self.config.long_query_words:
            signals.append('long_query')
        return signals

    # ── decomposition ───────────────────────────────────────────────────────
    def _decompose(self, text: str, signals: List[str]) -> List[str]:
        """Return the sub-queries to retrieve with, or [text] if the query
        cannot be split into meaningfully separate parts."""
        parts: List[str] = []
        include_original = False

        if len(_ENUM_MARKER_RE.findall(text)) >= 2:
            # Checked before '?'-splitting: "(1) A? (2) B?" must be split on
            # its markers, not on the question marks that happen to be there.
            lead_in, items = self._split_enumeration(text)
            parts = items
            # A lead-in may carry context shared by every item; keep the
            # full query in the retrieval set so that context isn't lost.
            include_original = bool(lead_in)
        elif 'multiple_questions' in signals:
            parts = self._split_questions(text)
        elif 'multi_part_conjunction' in signals:
            parts = self._split_conjunctions(text)

        # Split compound parts further ("A and how B?" inside a "?" segment).
        expanded: List[str] = []
        for part in parts:
            expanded.extend(self._split_conjunctions(part))
        parts = [p for p in expanded if self._is_substantive(p)]

        if len(parts) < 2:
            parts = []
            comparison = self._comparison_entities(text) if 'comparison' in signals else []
            if comparison:
                # Entity-only fragments are weak queries on their own, so
                # the original (which carries the shared context) stays.
                parts = comparison
                include_original = True

        if len(parts) < 2:
            return [text]

        ordered = ([text] if include_original else []) + parts
        return self._dedupe_and_cap(ordered)

    def _split_questions(self, text: str) -> List[str]:
        segments = re.findall(r'[^?]+\?|[^?]+$', text)
        return [s.strip() for s in segments if self._is_substantive(s.strip())]

    def _split_enumeration(self, text: str) -> Tuple[str, List[str]]:
        pieces = _ENUM_MARKER_RE.split(text)
        lead_in = pieces[0].strip()
        # A lead-in that only announces the list ("Answer the following:")
        # carries no retrieval context.
        if lead_in.endswith(':'):
            lead_in = ''
        items = [p.strip() for p in pieces[1:] if self._is_substantive(p.strip())]
        return lead_in, items

    @staticmethod
    def _split_conjunctions(text: str) -> List[str]:
        return [p.strip() for p in _CONJUNCTION_SPLIT_RE.split(text) if p and p.strip()]

    @staticmethod
    def _comparison_entities(text: str) -> List[str]:
        for pattern in _COMPARISON_ENTITY_PATTERNS:
            match = pattern.search(text)
            if match:
                a = match.group('a').strip(' ,;:')
                b = match.group('b').strip(' ,;:')
                if a and b and normalize_query(a) != normalize_query(b):
                    return QueryRouter._share_scope(a, b)
        return []

    @staticmethod
    def _share_scope(a: str, b: str) -> List[str]:
        """'BERT' / 'GPT-2 on pretraining objectives' -> both sides get the
        trailing scope, since 'on pretraining objectives' qualifies the
        comparison, not just the second entity."""
        m = _TRAILING_SCOPE_RE.match(b)
        if m and not _TRAILING_SCOPE_RE.match(a):
            entity_b, scope = m.group('entity'), m.group('scope')
            return [f'{a} {scope}', f'{entity_b} {scope}']
        return [a, b]

    def _is_substantive(self, fragment: str) -> bool:
        return len(fragment.split()) >= self.config.min_subquery_words

    def _dedupe_and_cap(self, queries: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for q in queries:
            key = normalize_query(q)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(q)
            if len(out) >= self.config.max_subqueries:
                break
        return out
