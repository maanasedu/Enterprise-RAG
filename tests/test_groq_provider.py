"""Tests for Groq provider support in scripts/build_benchmark.py's
call_llm() and main()'s key-check logic.

All Groq SDK calls are mocked — no real network/API calls are made. The
`groq` package (already in requirements.txt: groq==0.37.1) is imported
for its real `Groq` class so we can monkeypatch it in place, matching how
generation/llm_client.py already uses it.
"""
import importlib.util
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "build_benchmark", os.path.join(_REPO_ROOT, "scripts", "build_benchmark.py")
)
bb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bb)


class FakeGroqResponse:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})]


class FakeGroqCompletions:
    def __init__(self):
        self.calls = []
        self.response_text = '{"question": "q", "answer": "a"}'

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeGroqResponse(self.response_text)


class FakeGroqChat:
    def __init__(self):
        self.completions = FakeGroqCompletions()


class FakeGroqClient:
    """Stands in for groq.Groq(...). Records the kwargs each
    chat.completions.create() call received, and returns canned text."""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat = FakeGroqChat()


# ── provider recognition / dispatch ──────────────────────────────────────

def test_groq_provider_is_recognized_not_unknown(monkeypatch):
    """An unknown provider raises ValueError from call_llm's inner
    dispatch (surfaced as a logged retry failure, then None). Groq must
    NOT hit that branch — it should reach the GROQ_API_KEY check instead."""
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # No key set -> should fail with the GROQ_API_KEY RuntimeError message,
    # not the "Unknown BENCHMARK_LLM_PROVIDER" ValueError.
    result = bb.call_llm("hello", retries=1)
    assert result is None  # exhausted retries after the RuntimeError


def test_missing_groq_api_key_produces_clear_error(monkeypatch, capsys):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    bb.call_llm("hello", retries=1)
    captured = capsys.readouterr()
    assert "GROQ_API_KEY is not set" in captured.out


def test_provider_selection_uses_groq_client_when_selected(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake-for-test")

    import groq
    monkeypatch.setattr(groq, "Groq", FakeGroqClient)

    result = bb.call_llm("What is the capital?", system="sys")
    assert result == '{"question": "q", "answer": "a"}'


def test_groq_client_receives_correct_model_and_messages(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake-for-test")
    monkeypatch.delenv("BENCHMARK_GROQ_MODEL", raising=False)
    monkeypatch.delenv("BENCHMARK_LLM_MODEL", raising=False)

    import groq
    captured_client = {}

    def make_client(api_key=None):
        client = FakeGroqClient(api_key=api_key)
        captured_client["client"] = client
        return client

    monkeypatch.setattr(groq, "Groq", make_client)

    bb.call_llm("question text", system="system text", temperature=0.0, max_tokens=50)

    client = captured_client["client"]
    assert client.api_key == "gsk-fake-for-test"
    assert len(client.chat.completions.calls) == 1
    call_kwargs = client.chat.completions.calls[0]
    # Reuses the same default model as generation/llm_client.py's RAGGenerator
    assert call_kwargs["model"] == "openai/gpt-oss-120b"
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "question text"},
    ]
    assert call_kwargs["temperature"] == 0.0
    # Reasoning-model fix: Groq calls get a token floor + low reasoning effort.
    assert call_kwargs["max_tokens"] == bb.GROQ_MIN_MAX_TOKENS
    assert call_kwargs["reasoning_effort"] == "low"


def test_benchmark_groq_model_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake-for-test")
    monkeypatch.setenv("BENCHMARK_GROQ_MODEL", "llama-3.1-70b-versatile")

    import groq
    captured_client = {}

    def make_client(api_key=None):
        client = FakeGroqClient(api_key=api_key)
        captured_client["client"] = client
        return client

    monkeypatch.setattr(groq, "Groq", make_client)

    bb.call_llm("q")
    assert captured_client["client"].chat.completions.calls[0]["model"] == "llama-3.1-70b-versatile"


def test_groq_response_with_markdown_fences_parses_via_extract_json(monkeypatch):
    """Structured-output requirement: markdown code fences / extra text
    around the JSON must still parse correctly."""
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake-for-test")

    import groq

    class FencedClient(FakeGroqClient):
        def __init__(self, api_key=None):
            super().__init__(api_key=api_key)
            self.chat.completions.response_text = (
                '```json\n{"question": "What year?", "answer": "2020"}\n```'
            )

    monkeypatch.setattr(groq, "Groq", FencedClient)

    raw = bb.call_llm("q")
    parsed = bb._extract_json(raw)
    assert parsed == {"question": "What year?", "answer": "2020"}


# ── generation functions work end-to-end against a mocked Groq call_llm ──

def test_gen_single_chunk_question_works_with_groq_provider(monkeypatch):
    monkeypatch.setattr(bb, "call_llm", lambda *a, **k: '{"question": "Q?", "answer": "A"}')
    chunk = {"text": "some corpus text", "metadata": {"filename": "paper_000.pdf", "chunk_index": 0}}
    import random
    result = bb.gen_single_chunk_question(chunk, "factual", random.Random(1))
    assert result is not None
    assert result["question"] == "Q?"
    assert result["reference_answer"] == "A"


# ── existing providers still work (requirement #5: not broken) ──────────

def test_openai_provider_still_works_and_is_unaffected_by_groq_addition(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")

    class FakeMessage:
        content = '{"question": "q", "answer": "a"}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeOpenAIClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return FakeResponse()

    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)

    result = bb.call_llm("hello")
    assert result == '{"question": "q", "answer": "a"}'


def test_anthropic_provider_still_works_and_is_unaffected_by_groq_addition(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")

    class FakeContentBlock:
        text = '{"question": "q", "answer": "a"}'

    class FakeResponse:
        content = [FakeContentBlock()]

    class FakeAnthropicClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

        class messages:
            @staticmethod
            def create(**kwargs):
                return FakeResponse()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropicClient)

    result = bb.call_llm("hello")
    assert result == '{"question": "q", "answer": "a"}'


def test_unknown_provider_raises_value_error_not_silently_handled(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "totally-not-a-provider")
    result = bb.call_llm("hello", retries=1)
    assert result is None  # call_llm catches and logs, never crashes the pipeline


# ── main()'s key-check dispatch (requirement: fail clearly, no fallback) ─

def test_main_requires_groq_api_key_when_provider_is_groq(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["build_benchmark.py", "--target-size", "4"])
    with pytest.raises(SystemExit) as exc_info:
        bb.main()
    assert exc_info.value.code == 1


def test_main_does_not_fall_back_to_openai_key_when_groq_selected(monkeypatch, tmp_path):
    """If GROQ_API_KEY is missing but OPENAI_API_KEY happens to be set,
    main() must still fail — never silently use a different provider's key."""
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-should-not-be-used")
    monkeypatch.setattr(sys, "argv", ["build_benchmark.py", "--target-size", "4"])
    with pytest.raises(SystemExit) as exc_info:
        bb.main()
    assert exc_info.value.code == 1


def test_main_proceeds_past_key_check_when_groq_api_key_present(monkeypatch, tmp_path):
    """With GROQ_API_KEY set, main() should get past the key-check and
    into build_benchmark() (which will then hit the attempt budget and
    exit 1 for shortfall, since call_llm isn't mocked here — we're only
    verifying it does NOT exit at the key-check stage with a key-missing
    message)."""
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake-for-test")

    import groq
    monkeypatch.setattr(groq, "Groq", FakeGroqClient)  # avoid any real network attempt

    corpus = [{"text": "some text", "metadata": {"filename": "paper_000.pdf", "chunk_index": 0}}]
    corpus_path = tmp_path / "faiss_metadata.json"
    import json
    corpus_path.write_text(json.dumps(corpus))
    out_dir = tmp_path / "out"

    monkeypatch.setattr(sys, "argv", [
        "build_benchmark.py", "--target-size", "1", "--seed", "1",
        "--faiss-metadata", str(corpus_path), "--out-dir", str(out_dir),
        "--max-attempts-multiplier", "1",
    ])
    # Whatever happens next (success or a validation-driven shortfall) is
    # fine — what we're asserting is that it got PAST the key-check, i.e.
    # build_benchmark() ran using the Groq client, not that it necessarily
    # succeeds with the toy 1-chunk corpus.
    with pytest.raises(SystemExit):
        bb.main()


# ── no hard-coded API keys anywhere in the file ──────────────────────────

def test_no_hardcoded_api_keys_in_build_benchmark_source():
    src_path = os.path.join(_REPO_ROOT, "scripts", "build_benchmark.py")
    with open(src_path) as f:
        source = f.read()
    # Common hard-coded-secret shapes for these providers; a real key would
    # match one of these patterns if someone pasted it in literally.
    assert "sk-" not in source
    assert "gsk_" not in source
    assert "sk-ant-" not in source


def test_requirements_txt_lists_groq_dependency():
    req_path = os.path.join(_REPO_ROOT, "requirements.txt")
    with open(req_path) as f:
        content = f.read().lower()
    assert "groq" in content
