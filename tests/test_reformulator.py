"""Dimension 2: query reformulation (retrieval/reformulator.py)."""
import json
from types import SimpleNamespace

from retrieval.reformulator import HeuristicReformulator, LLMReformulator, keyword_query
from retrieval.router import QueryRouter, normalize_query


def test_keyword_query_strips_scaffolding_but_keeps_negation():
    assert keyword_query('What is the notice period for terminating the contract?') == \
        'notice period terminating contract'
    assert 'not' in keyword_query('Which employees are not eligible for the bonus?').split()


def test_keyword_query_returns_none_when_too_little_remains():
    assert keyword_query('What is it?') is None


def test_heuristic_reformulation_is_distinct_from_prior_queries():
    q = 'What is the notice period for terminating the contract?'
    out = HeuristicReformulator().reformulate(q, [q])
    assert out == ['notice period terminating contract']
    assert normalize_query(out[0]) != normalize_query(q)


def test_heuristic_returns_nothing_when_query_is_already_keyword_only():
    q = 'notice period terminating contract'
    assert HeuristicReformulator().reformulate(q, [q]) == []


def test_heuristic_reformulates_each_subquery_of_a_decomposed_question():
    q = 'What is the refund policy and how long does shipping take?'
    decision = QueryRouter().route(q)
    out = HeuristicReformulator().reformulate(q, [q, *decision.subqueries], decision)
    assert 'refund policy' in out and 'long shipping take' in out


def test_heuristic_respects_max_queries():
    q = 'What is the refund policy and how long does shipping take?'
    decision = QueryRouter().route(q)
    assert len(HeuristicReformulator(max_queries=1).reformulate(q, [q], decision)) == 1


# ── LLM reformulator (mocked client; never hits Groq) ────────────────────────
class FakeLLM:
    def __init__(self, content=None, raises=None):
        self.content, self.raises, self.calls = content, raises, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


Q = 'What is the notice period for terminating the contract?'


def test_llm_reformulator_parses_json_array_and_uses_one_call():
    llm = FakeLLM(json.dumps(['contract termination notice', 'how many days notice to end agreement']))
    out = LLMReformulator(llm, 'm').reformulate(Q, [Q])
    assert out == ['contract termination notice', 'how many days notice to end agreement']
    assert len(llm.calls) == 1
    assert llm.calls[0]['model'] == 'm'
    assert llm.calls[0]['temperature'] == 0.0


def test_llm_reformulator_accepts_fenced_json():
    llm = FakeLLM('```json\n["contract termination notice"]\n```')
    assert LLMReformulator(llm, 'm').reformulate(Q, [Q]) == ['contract termination notice']


def test_llm_reformulator_drops_queries_already_tried_and_caps_output():
    llm = FakeLLM(json.dumps([Q.upper(), 'a one', 'b two', 'c three', 'd four']))
    out = LLMReformulator(llm, 'm', max_queries=2).reformulate(Q, [Q])
    assert out == ['a one', 'b two']


def test_llm_reformulator_falls_back_to_heuristic_on_garbage():
    out = LLMReformulator(FakeLLM('not json at all'), 'm').reformulate(Q, [Q])
    assert out == ['notice period terminating contract']


def test_llm_reformulator_falls_back_to_heuristic_on_wrong_json_shape():
    out = LLMReformulator(FakeLLM('{"queries": ["x y"]}'), 'm').reformulate(Q, [Q])
    assert out == ['notice period terminating contract']


def test_llm_reformulator_falls_back_when_call_raises_and_does_not_retry():
    llm = FakeLLM(raises=RuntimeError('groq down'))
    out = LLMReformulator(llm, 'm').reformulate(Q, [Q])
    assert out == ['notice period terminating contract']
    assert len(llm.calls) == 1            # exactly one attempt, no retry loop


def test_llm_reformulator_falls_back_when_llm_only_repeats_prior_queries():
    out = LLMReformulator(FakeLLM(json.dumps([Q])), 'm').reformulate(Q, [Q])
    assert out == ['notice period terminating contract']
