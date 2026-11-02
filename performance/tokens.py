"""Provider-reported token usage. Never estimates: if the provider did not
report a number, there is no number."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TokenUsage:
    input: Optional[int] = None
    output: Optional[int] = None
    total: Optional[int] = None

    @property
    def complete(self) -> bool:
        """Both sides reported, so the call can be counted and priced."""
        return self.input is not None and self.output is not None


def _as_count(value) -> Optional[int]:
    """A non-negative whole number, or None. Rejects bool, NaN/inf, negatives,
    strings and mocks instead of coercing them into a plausible-looking count."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if math.isfinite(value) and value >= 0 and value == int(value):
            return int(value)
    return None


def _read(usage, *names: str) -> Optional[int]:
    for name in names:
        raw = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
        count = _as_count(raw)
        if count is not None:
            return count
    return None


def extract_usage(response) -> Optional[TokenUsage]:
    """Read ``response.usage`` (OpenAI/Groq: prompt/completion/total_tokens;
    Anthropic: input/output_tokens). None when the response carries no usable
    usage block."""
    usage = response.get('usage') if isinstance(response, Mapping) else getattr(response, 'usage', None)
    if usage is None:
        return None
    parsed = TokenUsage(
        input=_read(usage, 'prompt_tokens', 'input_tokens'),
        output=_read(usage, 'completion_tokens', 'output_tokens'),
        total=_read(usage, 'total_tokens'),
    )
    if parsed.input is None and parsed.output is None and parsed.total is None:
        return None
    return parsed


def aggregate_usage(calls: Sequence[Tuple[str, Optional[TokenUsage]]]) -> Optional[Dict]:
    """Sum usage over the LLM calls of one request.

    Only calls with both input and output reported are counted (a half-reported
    call cannot be priced). ``total`` is the provider's own total when given,
    else input + output. Returns None when no call reported usage. If some
    calls were made but did not report usage, ``incomplete`` is True: the
    totals are then a lower bound, not the full request.
    """
    counted = [(purpose, u) for purpose, u in calls if u is not None and u.complete]
    if not counted:
        return None

    def total_of(u: TokenUsage) -> int:
        return u.total if u.total is not None else u.input + u.output

    result: Dict = {
        'input': sum(u.input for _, u in counted),
        'output': sum(u.output for _, u in counted),
        'total': sum(total_of(u) for _, u in counted),
    }
    purposes = {purpose for purpose, _ in counted}
    if len(purposes) > 1:
        result['by_purpose'] = {
            purpose: {
                'input': sum(u.input for p, u in counted if p == purpose),
                'output': sum(u.output for p, u in counted if p == purpose),
                'total': sum(total_of(u) for p, u in counted if p == purpose),
            }
            for purpose in sorted(purposes)
        }
    if len(counted) < len(calls):
        result['incomplete'] = True
    return result
