"""Configurable LLM pricing and cost *estimation*.

Prices are data, not logic. Resolution order for the configured model:

  1. environment: LLM_PRICE_INPUT_PER_1M + LLM_PRICE_OUTPUT_PER_1M
     (both or neither), optional LLM_PRICE_CURRENCY (default USD)
  2. pricing file: LLM_PRICING_FILE, else config/llm_pricing.json, entry
     "<provider>/<model>" (e.g. "groq/openai/gpt-oss-120b")
  3. unset: cost is simply not estimated (never a guessed default)

The result is an ESTIMATE = reported tokens x configured price. It is not
billing data: free tiers, discounts, cached-token pricing, taxes and failed or
retried calls are not modelled.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

from retrieval.env_config import parse_float

ENV_INPUT = 'LLM_PRICE_INPUT_PER_1M'
ENV_OUTPUT = 'LLM_PRICE_OUTPUT_PER_1M'
ENV_CURRENCY = 'LLM_PRICE_CURRENCY'
ENV_FILE = 'LLM_PRICING_FILE'

DEFAULT_CURRENCY = 'USD'
DEFAULT_PRICING_FILE = Path(__file__).resolve().parent.parent / 'config' / 'llm_pricing.json'

SOURCE_ENV = 'env'
SOURCE_UNSET = 'unset'


def _valid_price(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}={value!r} is not a number')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name}={value!r} must be a finite, non-negative number')
    return value


@dataclass(frozen=True)
class PricingConfig:
    input_per_1m: Optional[float] = None
    output_per_1m: Optional[float] = None
    currency: str = DEFAULT_CURRENCY
    source: str = SOURCE_UNSET
    as_of: Optional[str] = None
    note: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.input_per_1m is not None and self.output_per_1m is not None

    def estimate_cost(self, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
        """Estimated cost, or None when pricing or either token count is missing.
        Zero tokens is a real value and costs 0.0 (it is not 'missing')."""
        if not self.available or input_tokens is None or output_tokens is None:
            return None
        cost = (input_tokens * self.input_per_1m + output_tokens * self.output_per_1m) / 1_000_000
        return round(cost, 8)

    def to_dict(self) -> Dict:
        return {
            'input_per_1m': self.input_per_1m, 'output_per_1m': self.output_per_1m,
            'currency': self.currency, 'source': self.source,
            'as_of': self.as_of, 'note': self.note,
        }

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, *,
                 provider: Optional[str] = None, model: Optional[str] = None) -> 'PricingConfig':
        env = os.environ if env is None else env
        raw_in, raw_out = env.get(ENV_INPUT), env.get(ENV_OUTPUT)
        blank_in = raw_in is None or raw_in.strip() == ''
        blank_out = raw_out is None or raw_out.strip() == ''
        currency = (env.get(ENV_CURRENCY) or '').strip()

        if blank_in != blank_out:
            raise ValueError(f'{ENV_INPUT} and {ENV_OUTPUT} must be set together (or both left unset)')
        if not blank_in:
            return cls(
                input_per_1m=_valid_price(ENV_INPUT, parse_float(ENV_INPUT, raw_in, None)),
                output_per_1m=_valid_price(ENV_OUTPUT, parse_float(ENV_OUTPUT, raw_out, None)),
                currency=currency or DEFAULT_CURRENCY, source=SOURCE_ENV,
            )
        return cls._from_file(env, provider, model, currency)

    @classmethod
    def _from_file(cls, env, provider, model, currency_override: str) -> 'PricingConfig':
        explicit = (env.get(ENV_FILE) or '').strip()
        path = Path(explicit) if explicit else DEFAULT_PRICING_FILE
        if not path.is_file():
            if explicit:
                raise ValueError(f'{ENV_FILE}={explicit!r}: file not found')
            return cls(currency=currency_override or DEFAULT_CURRENCY)
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            raise ValueError(f'pricing file {path}: not valid JSON ({e})') from None
        if not isinstance(data, dict) or not isinstance(data.get('models', {}), dict):
            raise ValueError(f'pricing file {path}: expected an object with a "models" object')

        currency = currency_override or str(data.get('currency') or DEFAULT_CURRENCY)
        if not provider or not model:
            return cls(currency=currency)          # cannot identify the model -> unpriced
        key = f'{provider}/{model}'
        entry = data.get('models', {}).get(key)
        if entry is None:
            return cls(currency=currency)
        if not isinstance(entry, dict):
            raise ValueError(f'pricing file {path}: entry {key!r} must be an object')
        return cls(
            input_per_1m=_valid_price(f'{key}.input_per_1m', entry.get('input_per_1m')),
            output_per_1m=_valid_price(f'{key}.output_per_1m', entry.get('output_per_1m')),
            currency=str(entry.get('currency') or currency),
            source=f'file:{path.name}:{key}',
            as_of=entry.get('as_of'),
            note=entry.get('note'),
        )
