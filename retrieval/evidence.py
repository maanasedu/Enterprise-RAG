"""Evidence evaluation over reranked chunks (Dimension 2).

Decides whether what retrieval returned is strong enough to generate from.
This module does *scoring and comparison only* — it does not route queries,
retrieve, reformulate or generate.

IMPORTANT — what the numbers are
--------------------------------
Cohere reranker scores are *ranking signals*: they order candidates for one
query, and their absolute values are not calibrated probabilities. A score
of 0.4 does not mean "40% likely relevant", and scores are not guaranteed
to be comparable across queries. So this module:

  * never labels a score a probability or confidence (see `EvidenceVerdict`),
  * never clamps, rescales or assumes a [0, 1] range,
  * compares raw scores against *configured* thresholds only.

The default threshold below is an UNCALIBRATED PLACEHOLDER. It is
deliberately permissive (it only rejects near-zero-relevance evidence) so
that enabling this layer does not silently change normal answers. The real
value must be chosen from Dimension 1 benchmark data: e.g. sweep
`min_top_score`, and pick the operating point trading refusals of
answerable questions against answers to unanswerable ones.

Unscored evidence
-----------------
When Cohere is unavailable (no key, outage) or the reranker is disabled,
`reranked` is False and `relevance_score` is a synthetic placeholder
derived from rank order. Thresholding those would be judging fabricated
numbers, so such evidence is reported as *not evaluated*, and
`fail_open_when_unscored` selects whether it is then allowed through
(default; identical to pre-Dimension-2 behavior) or refused.
"""
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from retrieval.env_config import parse_bool, parse_float, parse_int

# UNCALIBRATED placeholder — see module docstring. Not a probability.
DEFAULT_MIN_TOP_SCORE = 0.019681409
DEFAULT_TOP_K = 3

ENV_TOP_K = 'ADAPTIVE_EVIDENCE_TOP_K'
ENV_MIN_TOP_SCORE = 'ADAPTIVE_EVIDENCE_MIN_TOP_SCORE'
ENV_MIN_MEAN_SCORE = 'ADAPTIVE_EVIDENCE_MIN_MEAN_SCORE'
ENV_FAIL_OPEN = 'ADAPTIVE_EVIDENCE_FAIL_OPEN_WHEN_UNSCORED'


@dataclass(frozen=True)
class EvidenceConfig:
    """Threshold configuration. Scores are compared raw (not as probabilities).

    top_k:            how many of the best reranker scores are inspected.
    min_top_score:    the best score must be >= this.
    min_mean_score:   optional; the mean of the top_k scores must be >= this.
                      None disables the check.
    fail_open_when_unscored:
                      what to do when scores are not real reranker scores
                      (see module docstring). True = let the evidence
                      through; False = treat as insufficient.
    """
    top_k: int = DEFAULT_TOP_K
    min_top_score: float = DEFAULT_MIN_TOP_SCORE
    min_mean_score: Optional[float] = None
    fail_open_when_unscored: bool = True

    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError('top_k must be >= 1')
        for name in ('min_top_score', 'min_mean_score'):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number')

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> 'EvidenceConfig':
        env = os.environ if env is None else env
        return cls(
            top_k=parse_int(ENV_TOP_K, env.get(ENV_TOP_K), DEFAULT_TOP_K),
            min_top_score=parse_float(
                ENV_MIN_TOP_SCORE, env.get(ENV_MIN_TOP_SCORE), DEFAULT_MIN_TOP_SCORE
            ),
            min_mean_score=parse_float(ENV_MIN_MEAN_SCORE, env.get(ENV_MIN_MEAN_SCORE), None),
            fail_open_when_unscored=parse_bool(
                ENV_FAIL_OPEN, env.get(ENV_FAIL_OPEN), True
            ),
        )


# Verdict reasons (stable strings, safe to log / aggregate on).
REASON_OK = 'ok'
REASON_NO_EVIDENCE = 'no_evidence'
REASON_TOP_SCORE_BELOW = 'top_score_below_threshold'
REASON_MEAN_SCORE_BELOW = 'mean_score_below_threshold'
REASON_UNSCORED_ACCEPTED = 'unscored_accepted'
REASON_UNSCORED_REFUSED = 'unscored_refused'


@dataclass(frozen=True)
class EvidenceVerdict:
    sufficient: bool
    # False when the scores were not real reranker scores and so were not
    # compared against any threshold.
    evaluated: bool
    reason: str
    # Raw reranker scores as returned — ranking signals, NOT probabilities.
    top_score: Optional[float]
    mean_top_k_score: Optional[float]
    scores_considered: Tuple[float, ...]
    min_top_score: float
    min_mean_score: Optional[float]

    @property
    def retryable(self) -> bool:
        """True when evidence was genuinely judged weak, so a reformulated
        retrieval could plausibly help. Not retryable when nothing could be
        judged (a retry would produce the same unscoreable results)."""
        return self.evaluated and not self.sufficient

    def to_dict(self) -> Dict:
        return {
            'sufficient': self.sufficient,
            'evaluated': self.evaluated,
            'reason': self.reason,
            'top_score': self.top_score,
            'mean_top_k_score': self.mean_top_k_score,
            'scores_considered': list(self.scores_considered),
            'thresholds': {
                'min_top_score': self.min_top_score,
                'min_mean_score': self.min_mean_score,
            },
        }


def _usable_score(chunk: Dict) -> Optional[float]:
    if chunk.get('reranked') is not True:
        return None
    score = chunk.get('relevance_score')
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score = float(score)
    return score if math.isfinite(score) else None


class EvidenceEvaluator:
    def __init__(self, config: Optional[EvidenceConfig] = None):
        self.config = config or EvidenceConfig()

    def evaluate(self, chunks: List[Dict]) -> EvidenceVerdict:
        cfg = self.config

        def verdict(sufficient, evaluated, reason, top=None, mean=None, scores=()):
            return EvidenceVerdict(
                sufficient=sufficient,
                evaluated=evaluated,
                reason=reason,
                top_score=top,
                mean_top_k_score=mean,
                scores_considered=tuple(scores),
                min_top_score=cfg.min_top_score,
                min_mean_score=cfg.min_mean_score,
            )

        if not chunks:
            return verdict(False, True, REASON_NO_EVIDENCE)

        scores = [_usable_score(c) for c in chunks]
        if any(s is None for s in scores):
            # Synthetic (fallback / disabled-reranker) or missing scores:
            # do not compare them to a threshold.
            accept = cfg.fail_open_when_unscored
            return verdict(
                accept,
                False,
                REASON_UNSCORED_ACCEPTED if accept else REASON_UNSCORED_REFUSED,
            )

        # Do not rely on input order: take the k best raw scores.
        best = sorted(scores, reverse=True)[:cfg.top_k]
        top = best[0]
        mean = sum(best) / len(best)

        if top < cfg.min_top_score:
            return verdict(False, True, REASON_TOP_SCORE_BELOW, top, mean, best)
        if cfg.min_mean_score is not None and mean < cfg.min_mean_score:
            return verdict(False, True, REASON_MEAN_SCORE_BELOW, top, mean, best)
        return verdict(True, True, REASON_OK, top, mean, best)
