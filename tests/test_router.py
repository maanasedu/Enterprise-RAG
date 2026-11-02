"""Dimension 2: the deterministic query router (retrieval/router.py)."""
import pytest

from retrieval.router import Complexity, QueryRouter, RouterConfig, normalize_query

router = QueryRouter()


def test_simple_question_is_low_complexity_and_direct():
    q = 'What is the notice period for terminating the contract?'
    d = router.route(q)
    assert d.complexity == Complexity.LOW
    assert d.signals == ()
    assert d.subqueries == (q,)
    assert d.path == 'direct'
    assert not d.decomposed


@pytest.mark.parametrize('q', [
    'How does the transformer architecture work?',
    'Explain retrieval augmented generation.',
    'What different types of leave exist?',       # "different" alone is not a comparison
    'What are the pros and cons of BM25?',        # one topic, not a comparison of two things
    'why?',
])
def test_ordinary_questions_are_not_over_flagged(q):
    assert router.route(q).complexity == Complexity.LOW


def test_multiple_questions_are_split():
    d = router.route('What is attention? How does multi-head attention work?')
    assert d.complexity == Complexity.HIGH
    assert 'multiple_questions' in d.signals
    assert d.subqueries == ('What is attention?', 'How does multi-head attention work?')
    assert d.decomposed


def test_conjoined_questions_are_split():
    d = router.route('What is the refund policy and how long does shipping take?')
    assert d.complexity == Complexity.HIGH
    assert 'multi_part_conjunction' in d.signals
    assert d.subqueries == ('What is the refund policy', 'how long does shipping take?')


def test_comparison_keeps_original_and_adds_one_query_per_side():
    d = router.route('What is the difference between LoRA and full fine-tuning?')
    assert d.complexity == Complexity.HIGH
    assert 'comparison' in d.signals
    assert d.subqueries[0] == 'What is the difference between LoRA and full fine-tuning?'
    assert set(d.subqueries[1:]) == {'LoRA', 'full fine-tuning'}


def test_comparison_shares_trailing_scope_across_both_sides():
    d = router.route('Compare BERT and GPT-2 on pretraining objectives')
    assert d.subqueries[1:] == ('BERT on pretraining objectives', 'GPT-2 on pretraining objectives')


def test_vs_comparison():
    d = router.route('Redis vs Memcached for caching?')
    assert 'comparison' in d.signals
    assert d.subqueries[1:] == ('Redis for caching', 'Memcached for caching')


def test_enumerated_parts_split_on_markers_and_drop_announcing_lead_in():
    d = router.route(
        'Answer the following: (1) What is the rent? (2) What is the lease term? '
        '(3) Who is the landlord?'
    )
    assert 'enumerated_parts' in d.signals
    assert d.subqueries == ('What is the rent?', 'What is the lease term?', 'Who is the landlord?')
    assert not any('(2)' in s or '(3)' in s for s in d.subqueries)


def test_enumerated_parts_with_context_lead_in_keeps_original():
    q = 'For the 2023 lease, 1) what is the rent 2) what is the term'
    d = router.route(q)
    assert d.subqueries == (q, 'what is the rent', 'what is the term')


def test_multi_document_signal_is_high_but_not_decomposable():
    q = 'Summarize the main risks across the documents'
    d = router.route(q)
    assert d.complexity == Complexity.HIGH
    assert 'multi_document' in d.signals
    assert d.subqueries == (q,)          # nothing to split -> retrieve directly
    assert d.path == 'direct'


def test_long_query_signal_is_configurable():
    q = ' '.join(['word'] * 12)
    assert QueryRouter(RouterConfig(long_query_words=10)).route(q).complexity == Complexity.HIGH
    assert QueryRouter(RouterConfig(long_query_words=13)).route(q).complexity == Complexity.LOW


def test_subquery_cap_is_enforced():
    q = ' '.join(f'({i}) what is item number {i}?' for i in range(1, 9))
    d = QueryRouter(RouterConfig(max_subqueries=3)).route(q)
    assert len(d.subqueries) == 3


def test_subqueries_are_never_empty_and_deduplicated():
    d = router.route('What is X? What is X?')
    assert all(s.strip() for s in d.subqueries)
    assert len({normalize_query(s) for s in d.subqueries}) == len(d.subqueries)


def test_empty_query_does_not_crash():
    d = router.route('')
    assert d.complexity == Complexity.LOW
    assert len(d.subqueries) == 1


def test_decision_is_serializable_and_stable():
    d = router.route('What is attention? How does multi-head attention work?')
    assert d.to_dict() == {
        'complexity': 'high',
        'path': 'decomposed',
        'signals': ['multiple_questions'],
        'subqueries': ['What is attention?', 'How does multi-head attention work?'],
    }
    assert router.route('What is attention? How does multi-head attention work?') == d


@pytest.mark.parametrize('kwargs', [
    {'long_query_words': 0}, {'max_subqueries': 0}, {'min_subquery_words': 0},
])
def test_invalid_router_config_rejected(kwargs):
    with pytest.raises(ValueError):
        RouterConfig(**kwargs)
