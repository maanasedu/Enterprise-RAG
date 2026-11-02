"""Strict environment-variable parsing for the adaptive-routing settings.

A malformed value raises ValueError naming the variable, instead of being
silently replaced by a default — a mistyped threshold should stop the
service at startup, not quietly change refusal behavior. Unset or blank
variables fall back to the supplied default.
"""
import math
from typing import Optional

_TRUE = {'1', 'true', 'yes', 'on'}
_FALSE = {'0', 'false', 'no', 'off'}


def _blank(raw: Optional[str]) -> bool:
    return raw is None or raw.strip() == ''


def parse_bool(name: str, raw: Optional[str], default: bool) -> bool:
    if _blank(raw):
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f'{name}={raw!r} is not a boolean (use true/false)')


def parse_int(name: str, raw: Optional[str], default: int) -> int:
    if _blank(raw):
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f'{name}={raw!r} is not an integer') from None


def parse_float(name: str, raw: Optional[str], default: Optional[float]) -> Optional[float]:
    if _blank(raw):
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f'{name}={raw!r} is not a number') from None
    if not math.isfinite(value):
        raise ValueError(f'{name}={raw!r} must be a finite number')
    return value
