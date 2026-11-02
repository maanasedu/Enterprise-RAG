"""Dimension 2 evidence-threshold calibration: pure selection logic
(evaluation/threshold_calibration.py). No I/O, no network -- this is the
part of `scripts/calibrate_evidence_threshold.py` that can be fully tested
without a live Cohere key or the real corpus.
"""
import pytest

from evaluation.threshold_calibration import (
    ACCEPT_ALL_THRESHOLD,
    CalibrationRecord,
    candidate_thresholds,
    select_threshold,
    sweep_thresholds,
)


def rec(qid, status, score, evaluated=True, category=None, routing_path='direct'):
    return CalibrationRecord(qid, status, score, evaluated, category, routing_path)


def test_record_rejects_unknown_evidence_status():
    with pytest.raises(ValueError):
        rec('q', 'maybe', 0.5)


def test_record_requires_score_when_evaluated():
    with pytest.raises(ValueError):
        CalibrationRecord('q', 'sufficient', None, True)


def test_record_rejects_non_finite_score():
    with pytest.raises(ValueError):
        rec('q', 'sufficient', float('inf'))


def test_unevaluated_record_is_not_usable():
    r = rec('q', 'sufficient', None, evaluated=False)
    assert not r.usable


# ── candidate thresholds ──────────────────────────────────────────────────
def test_candidates_include_accept_all_and_every_observed_score():
    records = [rec('a', 'sufficient', 0.4), rec('b', 'insufficient', 0.1)]
    cands = candidate_thresholds(records)
    assert ACCEPT_ALL_THRESHOLD in cands
    assert 0.4 in cands and 0.1 in cands
    assert cands == sorted(cands)


def test_candidates_ignore_ambiguous_and_unusable_records():
    records = [
        rec('a', 'sufficient', 0.4),
        rec('b', 'ambiguous', 0.9),                 # excluded from objective
        rec('c', 'insufficient', 0.05, evaluated=False),  # unusable
    ]
    assert candidate_thresholds(records) == [0.0, 0.4, 1.4]  # 1.4 = 0.4 + 1.0 sentinel


# ── sweep ────────────────────────────────────────────────────────────────
def test_sweep_counts_accept_reject_correctly_on_both_sides():
    records = [
        rec('s1', 'sufficient', 0.9), rec('s2', 'sufficient', 0.3),
        rec('i1', 'insufficient', 0.6), rec('i2', 'insufficient', 0.1),
    ]
    points = {p.threshold: p for p in sweep_thresholds(records)}

    # threshold 0.0: accept everything.
    p0 = points[0.0]
    assert (p0.accepted_sufficient, p0.refused_sufficient) == (2, 0)
    assert (p0.accepted_insufficient, p0.refused_insufficient) == (2, 0)

    # threshold 0.6: accept score >= 0.6 -> s1(0.9) and i1(0.6) accepted.
    p6 = points[0.6]
    assert (p6.accepted_sufficient, p6.refused_sufficient) == (1, 1)
    assert (p6.accepted_insufficient, p6.refused_insufficient) == (1, 1)


def test_sweep_rates_and_accuracy():
    records = [rec('s1', 'sufficient', 0.9), rec('i1', 'insufficient', 0.1)]
    p = next(p for p in sweep_thresholds(records) if p.threshold == 0.9)
    assert p.false_refusal_rate == 0.0
    assert p.false_accept_rate == 0.0
    assert p.accuracy == 1.0


def test_sweep_rate_is_none_when_class_absent():
    records = [rec('s1', 'sufficient', 0.9)]
    p = sweep_thresholds(records)[0]
    assert p.false_accept_rate is None    # no insufficient items at all


# ── selection rule ───────────────────────────────────────────────────────
def test_perfect_separation_selects_the_gap():
    # sufficient scores well above insufficient scores -> the ideal
    # threshold sits strictly between them and rejects every insufficient
    # item at zero false-refusal cost.
    records = [
        rec('s1', 'sufficient', 0.9), rec('s2', 'sufficient', 0.8),
        rec('i1', 'insufficient', 0.05), rec('i2', 'insufficient', 0.02),
    ]
    result = select_threshold(records, placeholder_threshold=0.05)
    assert result.selected_threshold == 0.8          # largest score that still keeps both sufficient
    assert result.selected_point.refused_sufficient == 0
    assert result.selected_point.refused_insufficient == 2   # both insufficient rejected
    assert result.replace_placeholder is True


def test_overlapping_scores_never_produces_a_false_refusal():
    # One insufficient item scores HIGHER than the lowest sufficient item --
    # no threshold can reject it without also refusing a sufficient item.
    records = [
        rec('s1', 'sufficient', 0.9), rec('s2', 'sufficient', 0.2),
        rec('i1', 'insufficient', 0.5),  # sits between s1 and s2
        rec('i2', 'insufficient', 0.05),
    ]
    result = select_threshold(records, placeholder_threshold=0.05)
    assert result.selected_point.refused_sufficient == 0     # hard constraint holds
    assert result.selected_threshold == 0.2                  # can't exceed the lower sufficient score
    assert result.selected_point.refused_insufficient == 1   # only i2 (0.05 < 0.2) is rejected
    assert result.selected_point.accepted_insufficient == 1  # i1 unavoidably accepted


def test_ties_break_toward_the_largest_threshold():
    # Two candidate thresholds both achieve (0 false refusals, 1 rejected
    # insufficient); the larger one must be selected.
    records = [
        rec('s1', 'sufficient', 0.9),
        rec('i1', 'insufficient', 0.1), rec('i2', 'insufficient', 0.2),
    ]
    # thresholds in (0.2, 0.9] all reject both i1,i2 -> better than tie case;
    # construct a genuine tie: only one insufficient item, two thresholds
    # both reject it.
    records = [
        rec('s1', 'sufficient', 0.9),
        rec('i1', 'insufficient', 0.1),
    ]
    result = select_threshold(records, placeholder_threshold=0.05)
    # Any threshold in (0.1, 0.9] rejects i1 and keeps s1 -> the rule must
    # pick the largest representable one, i.e. 0.9 (s1's own score).
    assert result.selected_threshold == 0.9


def test_no_threshold_can_separate_perfectly_overlapping_scores():
    records = [rec('s1', 'sufficient', 0.5), rec('i1', 'insufficient', 0.5)]
    result = select_threshold(records, placeholder_threshold=0.05)
    assert result.selected_point.refused_sufficient == 0
    assert result.selected_point.accepted_insufficient == 1  # unavoidable
    assert result.selected_threshold == 0.5                  # ties at s1's own score


def test_placeholder_is_not_replaced_when_selection_matches_it():
    records = [rec('s1', 'sufficient', 0.05), rec('i1', 'insufficient', 0.0)]
    result = select_threshold(records, placeholder_threshold=0.05)
    assert result.selected_threshold == 0.05
    assert result.replace_placeholder is False


def test_ambiguous_records_are_excluded_from_objective_but_reported():
    records = [
        rec('s1', 'sufficient', 0.9), rec('i1', 'insufficient', 0.1),
        rec('a1', 'ambiguous', 0.4), rec('a2', 'ambiguous', 0.7),
    ]
    result = select_threshold(records, placeholder_threshold=0.05)
    assert result.n_ambiguous == 2
    assert set(result.ambiguous_scores) == {0.4, 0.7}
    # ambiguous scores never influenced the objective:
    assert result.selected_point.n_sufficient == 1
    assert result.selected_point.n_insufficient == 1


def test_unusable_records_are_excluded_and_counted():
    records = [
        rec('s1', 'sufficient', 0.9),
        rec('i1', 'insufficient', 0.1),
        rec('x1', 'sufficient', None, evaluated=False),
    ]
    result = select_threshold(records, placeholder_threshold=0.05)
    assert result.n_unusable == 1
    assert result.selected_point.n_sufficient == 1  # x1 not counted


def test_selection_requires_at_least_one_usable_record():
    with pytest.raises(ValueError):
        select_threshold([], placeholder_threshold=0.05)
    with pytest.raises(ValueError):
        select_threshold([rec('a', 'ambiguous', 0.5)], placeholder_threshold=0.05)


def test_selection_requires_at_least_one_sufficient_record():
    with pytest.raises(ValueError):
        select_threshold([rec('i1', 'insufficient', 0.1)], placeholder_threshold=0.05)


def test_result_serializes_to_dict_with_expected_shape():
    records = [rec('s1', 'sufficient', 0.9), rec('i1', 'insufficient', 0.1)]
    d = select_threshold(records, placeholder_threshold=0.05).to_dict()
    assert set(d) == {
        'selected_threshold', 'rule', 'counts', 'selected_point', 'sweep',
        'ambiguous_diagnostic_scores', 'placeholder_threshold', 'replace_placeholder',
    }
    assert d['counts']['n_calibration_questions'] == 2


def test_module_never_clamps_or_rescales_scores_outside_unit_interval():
    # Scores far outside [0, 1] must sweep/select exactly like any other
    # scale -- these are ranking signals, not probabilities.
    records = [rec('s1', 'sufficient', 12.5), rec('i1', 'insufficient', -3.0)]
    result = select_threshold(records, placeholder_threshold=0.05)
    assert result.selected_threshold == 12.5
    assert result.selected_point.refused_insufficient == 1
