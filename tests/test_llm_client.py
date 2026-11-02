"""Regression tests for generation/llm_client.py RAGGenerator (Fix #4:
context is assembled from whole chunks within a budget, not raw-sliced)."""
import os
import pytest

pytest.importorskip('groq')


@pytest.fixture(autouse=True)
def _fake_groq_key(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'fake-key-for-test')


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMsg(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class RecordingCompletions:
    def __init__(self, answer='This is a fine generated answer.'):
        self.answer = answer
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeResponse(self.answer)


def _make_generator():
    from generation.llm_client import RAGGenerator
    gen = RAGGenerator()
    completions = RecordingCompletions()
    gen.client = type('FakeClient', (), {'chat': type('FakeChat', (), {'completions': completions})()})()
    return gen, completions


def test_context_never_contains_old_truncation_marker():
    gen, completions = _make_generator()
    chunks = [{'text': 'A' * 4000}, {'text': 'B' * 4000}, {'text': 'C' * 100}]
    result = gen.generate('question?', chunks)

    user_message = completions.last_kwargs['messages'][1]['content']
    assert '[Context truncated]' not in user_message  # old raw-slicing marker is gone
    assert result['context_chunks_included'] == 1
    assert result['context_truncated'] is True
    assert 'A' * 4000 in result['context']
    assert 'B' not in result['context']


def test_all_chunks_fit_when_small():
    gen, _ = _make_generator()
    chunks = [{'text': 'short chunk one'}, {'text': 'short chunk two'}]
    result = gen.generate('question?', chunks)
    assert result['context_chunks_included'] == 2
    assert result['context_truncated'] is False


def test_empty_chunks_still_generates():
    gen, _ = _make_generator()
    result = gen.generate('question?', [])
    assert result['answer']
    assert result['context'] == 'No context available.'
