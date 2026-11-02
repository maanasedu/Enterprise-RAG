"""Dimension 2 -- evidence-threshold calibration (selection logic only).

This module is pure: no file I/O, no network, no Cohere/Groq client. It
takes already-collected top reranker scores (one per calibration question,
produced by `scripts/calibrate_evidence_threshold.py` against the real
`retrieval/evidence.py` + `retrieval/adaptive.py` pipeline) and picks a
`min_top_score` for `retrieval.evidence.EvidenceConfig`.

Reranker scores are ranking signals, not probabilities (see
`retrieval/evidence.py`). This module never assumes a [0, 1] range, never
treats a score as a probability, and never clamps or rescales scores --
thresholds are only ever compared against the same raw scores the production
`EvidenceEvaluator` computes.

Calibration objective
----------------------
Each calibration item has a benchmark `evidence_status`:

  sufficient   -- corpus contains the answer. The evaluator SHOULD accept.
                  Accepting is correct; refusing is a *false refusal*
                  (a real answer withheld from the user).
  insufficient -- corpus does not contain the answer (no gold evidence at
                  all). The evaluator SHOULD refuse. Refusing is correct;
                  accepting is a *false accept* (answering without support,
                  the precursor to a hallucinated answer).
  ambiguous    -- question admits multiple interpretations; retrieval may
                  still surface topically-relevant evidence. This is a
                  genuinely different failure mode (the desired behavior is
                  "ask a clarifying question", not "accept" or "refuse") that
                  the current binary evidence gate does not model -- see
                  `evaluation/benchmark_schema.py::EVIDENCE_STATUS_EXPECTED_BEHAVIOR`.
                  These items are therefore EXCLUDED from threshold
                  selection and reported separately as a diagnostic only.

Selection rule (documented, deterministic)
-------------------------------------------
Among the candidate thresholds, select the LARGEST threshold that produces
ZERO false refusals on the calibration set's `sufficient` items -- i.e. the
selection *hard-constrains* on "accepting sufficient evidence" /
"avoiding unnecessary refusals" (both requirements read as the same
constraint: never refuse an answerable question observed at calibration
time), and, subject to that constraint, maximizes how many `insufficient`
items are correctly rejected ("rejecting insufficient evidence").

This is a conservative rule by design: on a small calibration set (dozens,
not thousands, of questions) a rule that also lets *some* false refusals
through in exchange for better separation is easy to overfit to sampling
noise in exactly the class (false refusals) this system is meant to avoid.
The tradeoff is explicit and inspectable: `sweep_thresholds` returns the
full curve, so a different rule can be applied to the same collected scores
without re-running retrieval.

Ties (multiple thresholds with the same false-refusal and false-accept
counts) are broken toward the LARGEST such threshold, which rejects the
most insufficient-evidence questions without cost, on the current data.

Threshold `0.0` (accept everything) is always a valid candidate, so a
threshold is always selected -- there is no "calibration failed" case
baked into `select_threshold`; poor separation shows up as a threshold
close to (or at) `0.0` and a low insufficient-rejection rate, which
`format_report` surfaces plainly rather than hiding.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

SUFFICIENT = 'sufficient'
INSUFFICIENT = 'insufficient'
AMBIGUOUS = 'ambiguous'
OBJECTIVE_STATUSES = (SUFFICIENT, INSUFFICIENT)  # ambiguous is diagnostic-only

# The floor of the sweep -- "accept everything" is always a legal choice,
# and the score placeholder in retrieval/evidence.py is documented as
# non-negative, so 0.0 (rather than -inf, which EvidenceConfig also
# rejects as non-finite) is the natural "accept everything" sentinel.
ACCEPT_ALL_THRESHOLD = 0.0


@dataclass(frozen=True)
class CalibrationRecord:
    """One calibration question's collected top reranker score.

    `top_score` is the raw score `EvidenceEvaluator` would compare against
    a threshold (see `EvidenceVerdict.top_score`). `evaluated=False` means
    Cohere did not actually score this question (no key / API error /
    reranker disabled) -- see `retrieval/evidence.py`'s docstring on
    synthetic fallback scores. Such records carry no calibration signal
    and are excluded from every sweep/selection computation, but are kept
    in the report for visibility (a run with many of these is not a valid
    calibration run).
    """
    question_id: str
    evidence_status: str
    top_score: Optional[float]
    evaluated: bool
    category: Optional[str] = None
    routing_path: Optional[str] = None  # 'direct' | 'decomposed' | None

    def __post_init__(self):
        if self.evidence_status not in (SUFFICIENT, INSUFFICIENT, AMBIGUOUS):
            raise ValueError(f'unknown evidence_status: {self.evidence_status!r}')
        if self.evaluated and self.top_score is None:
            raise ValueError('evaluated=True records must carry a top_score')
        if self.evaluated and not math.isfinite(self.top_score):
            raise ValueError(f'top_score must be finite, got {self.top_score!r}')

    @property
    def usable(self) -> bool:
        """True iff this record carries real calibration signal."""
        return self.evaluated and self.top_score is not None


def _usable(records: Sequence[CalibrationRecord]) -> List[CalibrationRecord]:
    return [r for r in records if r.usable]


def _objective(records: Sequence[CalibrationRecord]) -> List[CalibrationRecord]:
    return [r for r in _usable(records) if r.evidence_status in OBJECTIVE_STATUSES]


def candidate_thresholds(records: Sequence[CalibrationRecord]) -> List[float]:
    """Every threshold that can change at least one accept/reject decision,
    plus the accept-everything floor. Exhaustive (not a fixed grid): between
    any two adjacent candidates every item's decision is identical, so no
    candidate threshold is skipped over.
    """
    scores = sorted({r.top_score for r in _objective(records)})
    candidates = {ACCEPT_ALL_THRESHOLD}
    candidates.update(scores)
    # A hair above the top score, so "reject everything" is representable
    # in the swept curve (never selected by the rule above, but needed for
    # a complete, inspectable curve in the report).
    if scores:
        candidates.add(scores[-1] + 1.0)
    return sorted(candidates)


@dataclass(frozen=True)
class ThresholdSweepPoint:
    threshold: float
    # Counts over the objective (sufficient + insufficient) items only.
    accepted_sufficient: int
    refused_sufficient: int      # false refusals -- the "unnecessary refusal" cost
    accepted_insufficient: int   # false accepts -- the hallucination-risk cost
    refused_insufficient: int

    @property
    def n_sufficient(self) -> int:
        return self.accepted_sufficient + self.refused_sufficient

    @property
    def n_insufficient(self) -> int:
        return self.accepted_insufficient + self.refused_insufficient

    @property
    def false_refusal_rate(self) -> Optional[float]:
        return self.refused_sufficient / self.n_sufficient if self.n_sufficient else None

    @property
    def false_accept_rate(self) -> Optional[float]:
        return self.accepted_insufficient / self.n_insufficient if self.n_insufficient else None

    @property
    def accuracy(self) -> Optional[float]:
        total = self.n_sufficient + self.n_insufficient
        if not total:
            return None
        correct = self.accepted_sufficient + self.refused_insufficient
        return correct / total

    def to_dict(self) -> Dict:
        return {
            'threshold': self.threshold,
            'accepted_sufficient': self.accepted_sufficient,
            'refused_sufficient': self.refused_sufficient,
            'accepted_insufficient': self.accepted_insufficient,
            'refused_insufficient': self.refused_insufficient,
            'false_refusal_rate': self.false_refusal_rate,
            'false_accept_rate': self.false_accept_rate,
            'accuracy': self.accuracy,
        }


def sweep_thresholds(records: Sequence[CalibrationRecord],
                     thresholds: Optional[Sequence[float]] = None
                     ) -> List[ThresholdSweepPoint]:
    """Evaluate every candidate threshold against the objective records.

    An item is "accepted" at threshold t iff top_score >= t (matches
    EvidenceEvaluator's `top < cfg.min_top_score` -> insufficient check).
    """
    objective = _objective(records)
    thresholds = candidate_thresholds(records) if thresholds is None else sorted(thresholds)

    points = []
    for t in thresholds:
        acc_suf = ref_suf = acc_ins = ref_ins = 0
        for r in objective:
            accepted = r.top_score >= t
            if r.evidence_status == SUFFICIENT:
                acc_suf += accepted
                ref_suf += not accepted
            else:
                acc_ins += accepted
                ref_ins += not accepted
        points.append(ThresholdSweepPoint(t, acc_suf, ref_suf, acc_ins, ref_ins))
    return points


@dataclass(frozen=True)
class CalibrationResult:
    selected_threshold: float
    rule: str
    n_calibration_questions: int
    n_sufficient: int
    n_insufficient: int
    n_ambiguous: int
    n_unusable: int              # evaluated=False (Cohere fallback) records, excluded throughout
    selected_point: ThresholdSweepPoint
    sweep: Tuple[ThresholdSweepPoint, ...]
    ambiguous_scores: Tuple[Optional[float], ...]
    replace_placeholder: bool
    placeholder_threshold: float

    def to_dict(self) -> Dict:
        return {
            'selected_threshold': self.selected_threshold,
            'rule': self.rule,
            'counts': {
                'n_calibration_questions': self.n_calibration_questions,
                'n_sufficient': self.n_sufficient,
                'n_insufficient': self.n_insufficient,
                'n_ambiguous_excluded_from_objective': self.n_ambiguous,
                'n_unusable_excluded_no_real_cohere_score': self.n_unusable,
            },
            'selected_point': self.selected_point.to_dict(),
            'sweep': [p.to_dict() for p in self.sweep],
            'ambiguous_diagnostic_scores': list(self.ambiguous_scores),
            'placeholder_threshold': self.placeholder_threshold,
            'replace_placeholder': self.replace_placeholder,
        }


SELECTION_RULE_DESCRIPTION = (
    "Select the LARGEST threshold with zero false refusals on calibration "
    "items labeled evidence_status='sufficient' (hard constraint: never "
    "refuse evidence the benchmark says is sufficient / avoid unnecessary "
    "refusals). Subject to that constraint, maximize the number of "
    "evidence_status='insufficient' items correctly refused (reject "
    "insufficient evidence). Ties broken toward the largest threshold. "
    "evidence_status='ambiguous' items are excluded from this objective "
    "(see module docstring) and reported separately as a diagnostic."
)


def select_threshold(records: Sequence[CalibrationRecord],
                     placeholder_threshold: float) -> CalibrationResult:
    objective = _objective(records)
    if not objective:
        raise ValueError(
            'No usable calibration records with evidence_status in '
            f'{OBJECTIVE_STATUSES}; cannot calibrate a threshold.'
        )
    if not any(r.evidence_status == SUFFICIENT for r in objective):
        raise ValueError(
            "No usable 'sufficient' calibration records; the zero-false-refusal "
            "constraint is meaningless without at least one."
        )

    sweep = sweep_thresholds(records)
    # Points that satisfy the hard constraint; ACCEPT_ALL_THRESHOLD (0.0)
    # always yields zero false refusals, so this is never empty.
    zero_false_refusal = [p for p in sweep if p.refused_sufficient == 0]

    best = max(
        zero_false_refusal,
        key=lambda p: (p.refused_insufficient, p.threshold),
    )

    n_sufficient = sum(1 for r in objective if r.evidence_status == SUFFICIENT)
    n_insufficient = sum(1 for r in objective if r.evidence_status == INSUFFICIENT)
    n_ambiguous = sum(1 for r in records if r.evidence_status == AMBIGUOUS)
    n_unusable = sum(1 for r in records if not r.usable)
    ambiguous_scores = tuple(
        r.top_score for r in records if r.evidence_status == AMBIGUOUS and r.usable
    )

    return CalibrationResult(
        selected_threshold=best.threshold,
        rule=SELECTION_RULE_DESCRIPTION,
        n_calibration_questions=len(records),
        n_sufficient=n_sufficient,
        n_insufficient=n_insufficient,
        n_ambiguous=n_ambiguous,
        n_unusable=n_unusable,
        selected_point=best,
        sweep=tuple(sweep),
        ambiguous_scores=ambiguous_scores,
        replace_placeholder=(best.threshold != placeholder_threshold),
        placeholder_threshold=placeholder_threshold,
    )
