"""Regression tests for generation/context_budget.py (Fix #4: safe context
budgeting instead of raw character slicing like context[:3000]/[:6000])."""
from generation.context_budget import build_context


def test_empty_chunks():
    assert build_context([], 100) == ('', 0, False)


def test_all_chunks_fit_within_budget():
    chunks = [{'text': 'short one'}, {'text': 'short two'}]
    ctx, n, truncated = build_context(chunks, max_chars=1000)
    assert n == 2
    assert not truncated
    assert '[Source 1]' in ctx and '[Source 2]' in ctx
    assert 'short one' in ctx and 'short two' in ctx


def test_lower_ranked_chunk_dropped_whole_not_sliced():
    chunks = [{'text': 'A' * 50}, {'text': 'B' * 50}, {'text': 'C' * 50}]
    ctx, n, truncated = build_context(chunks, max_chars=70)
    assert n == 1
    assert truncated
    # The included chunk must be complete, never a partial slice.
    assert 'A' * 50 in ctx
    assert 'B' not in ctx and 'C' not in ctx


def test_best_ranked_chunk_kept_whole_even_if_it_alone_exceeds_budget():
    chunks = [{'text': 'X' * 500}, {'text': 'Y' * 10}]
    ctx, n, truncated = build_context(chunks, max_chars=50)
    assert n == 1
    assert 'X' * 500 in ctx  # not cut short
    assert truncated


def test_missing_text_key_does_not_crash():
    chunks = [{'no_text_key': True}]
    ctx, n, truncated = build_context(chunks, max_chars=100)
    assert n == 1
    assert not truncated
