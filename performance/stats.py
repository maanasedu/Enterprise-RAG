"""Small statistics helpers (pure Python, no numpy)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

MIN_N_FOR_P95 = 20
MIN_N_FOR_P99 = 100


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """The q-th percentile (0-100) using linear interpolation between order
    statistics (the same definition as numpy.percentile's default).

    Returns None for an empty sequence. Non-finite values are rejected rather
    than silently skewing the result.
    """
    if not 0 <= q <= 100:
        raise ValueError(f'percentile q must be within [0, 100], got {q}')
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if not all(math.isfinite(v) for v in ordered):
        raise ValueError('percentile requires finite values')
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(value, digits)


def summarize(values: Sequence[float], digits: int = 2) -> Dict:
    """n / mean / min / max / p50 / p95 / p99. Every statistic is None when
    there are no values (n == 0): an empty sample has no latency, not 0 ms."""
    values = [float(v) for v in values]
    return {
        'n': len(values),
        'mean': _round(mean(values), digits),
        'min': _round(min(values), digits) if values else None,
        'max': _round(max(values), digits) if values else None,
        'p50': _round(percentile(values, 50), digits),
        'p95': _round(percentile(values, 95), digits),
        'p99': _round(percentile(values, 99), digits),
    }


def sample_size_warnings(n: int) -> List[str]:
    """Plain-language caution about how much a sample of size n can say."""
    warnings: List[str] = []
    if n == 0:
        warnings.append('no successful requests: no latency statistics')
        return warnings
    if n < MIN_N_FOR_P95:
        warnings.append(
            f'n={n} < {MIN_N_FOR_P95}: P95 is poorly estimated (it is close to the max)')
    if n < MIN_N_FOR_P99:
        warnings.append(f'n={n} < {MIN_N_FOR_P99}: P99 is not meaningful at this sample size')
    return warnings
