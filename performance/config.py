"""D5-Lite settings, read once at startup (strict parsing, like the other
dimensions' configs)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

from performance.pricing import PricingConfig
from retrieval.env_config import parse_bool

ENV_METADATA_ENABLED = 'PERFORMANCE_METADATA_ENABLED'


@dataclass(frozen=True)
class PerformanceConfig:
    # False (default): `performance` is added to a /query response only when
    # the request sets include_performance=true, so the default response is
    # byte-for-byte what it was before D5-Lite. True: always added.
    include_in_response: bool = False
    pricing: PricingConfig = PricingConfig()

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, *,
                 provider: Optional[str] = None, model: Optional[str] = None) -> 'PerformanceConfig':
        env = os.environ if env is None else env
        return cls(
            include_in_response=parse_bool(
                ENV_METADATA_ENABLED, env.get(ENV_METADATA_ENABLED), False),
            pricing=PricingConfig.from_env(env, provider=provider, model=model),
        )
