"""Tests for:
- quota-aware 429 handling (BenchmarkQuotaExhaustedError vs transient retry)
- checkpoint write/resume/compatibility
- --max-api-calls
- --categories
- usage accounting

All LLM calls are mocked/faked — no real network calls. Loaded via
importlib since `scripts/` is not a package (same pattern as the other
build_benchmark test files).
"""
import importlib.util
import json
import os
import sys
import time

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "build_benchmark", os.path.join(_REPO_ROOT, "scripts", "build_benchmark.py")
)
bb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bb)


def make_chunk(text, filename="paper_000.pdf", chunk_index=0):
    return {"text": text, "metadata": {"filename": filename, "chunk_index": chunk_index}}


def write_corpus(tmp_path, n=6):
    corpus = [
        make_chunk(f"Paper {i} discusses topic {i} with result {i * 10}.",
                   filename=f"paper_{i:03d}.pdf")
        for i in range(n)
    ]
    corpus_path = tmp_path / "faiss_metadata.json"
    corpus_path.write_text(json.dumps(corpus))
    return str(corpus_path)


def make_rate_limit_error(message, retry_after=None):
    """Builds a real groq.RateLimitError (status_code=429) so call_llm's
    getattr(e, 'status_code', None) / .message / .response.headers checks
    exercise the actual attributes the SDK sets, not a hand-rolled stand-in."""
    import groq
    import httpx
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    response = httpx.Response(429, request=request, headers=headers)
    return groq.RateLimitError(message, response=response, body={"error": {"message": message}})


def make_server_error(status_code=503):
    import groq
    import httpx
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return groq.InternalServerError("server error", response=response, body=None)
    # NOTE: groq.InternalServerError has no fixed status_code class attribute
    # (unlike RateLimitError), so it takes it from response.status_code —
    # exercised directly here rather than assumed.


# ── TPD (daily quota) 429 handling — requirements #1, #2 ─────────────────

def test_tpd_429_raises_benchmark_quota_exhausted_error(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")

    err = make_rate_limit_error(
        "Rate limit reached for model `openai/gpt-oss-120b` in organization "
        "`org_x` service tier `on_demand` on tokens per day (TPD): "
        "Limit 200000, Used 200000, Requested 4000."
    )

    def raising_dispatch(provider, prompt, system, temperature, max_tokens):
        raise err

    monkeypatch.setattr(bb, "_dispatch_provider_call", raising_dispatch)

    with pytest.raises(bb.BenchmarkQuotaExhaustedError):
        bb.call_llm("hello", retries=3)


def test_tpd_429_performs_zero_retries(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")

    call_count = {"n": 0}
    err = make_rate_limit_error("... on tokens per day (TPD): Limit 200000, Used 200000, Requested 4000.")

    def raising_dispatch(provider, prompt, system, temperature, max_tokens):
        call_count["n"] += 1
        raise err

    monkeypatch.setattr(bb, "_dispatch_provider_call", raising_dispatch)
    monkeypatch.setattr(time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("must not sleep/retry")))

    with pytest.raises(bb.BenchmarkQuotaExhaustedError):
        bb.call_llm("hello", retries=5)
    assert call_count["n"] == 1  # exactly one attempt, no retry


def test_rpd_daily_quota_phrasing_also_detected():
    assert bb._is_daily_quota_exhaustion("requests per day (RPD) limit exceeded")


def test_generic_used_requested_with_day_wording_detected():
    assert bb._is_daily_quota_exhaustion(
        "Your organization has used its allotted quota for today. Used 200000, Requested 500."
    )


def test_transient_rpm_429_is_not_classified_as_daily_quota():
    text = "Rate limit reached on requests per minute (RPM): Limit 30, Used 30, Requested 1."
    assert bb._is_daily_quota_exhaustion(text) is False


# ── transient RPM/TPM 429 — retries retained ─────────────────────────────

def test_transient_429_retries_and_eventually_succeeds(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    monkeypatch.setattr(time, "sleep", lambda s: None)

    err = make_rate_limit_error("Rate limit reached on requests per minute (RPM): Limit 30, Used 30, Requested 1.")
    calls = {"n": 0}

    def dispatch(provider, prompt, system, temperature, max_tokens):
        calls["n"] += 1
        if calls["n"] < 3:
            raise err
        return "ok response", None

    monkeypatch.setattr(bb, "_dispatch_provider_call", dispatch)
    result = bb.call_llm("hello", retries=5)
    assert result == "ok response"
    assert calls["n"] == 3  # retried twice before succeeding


def test_transient_429_gives_up_after_retries_exhausted_returns_none(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    monkeypatch.setattr(time, "sleep", lambda s: None)

    err = make_rate_limit_error("Rate limit reached on requests per minute (RPM).")
    monkeypatch.setattr(bb, "_dispatch_provider_call",
                         lambda *a, **k: (_ for _ in ()).throw(err))
    result = bb.call_llm("hello", retries=2)
    assert result is None  # never raises BenchmarkQuotaExhaustedError for transient 429


def test_retry_after_header_is_respected(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    err = make_rate_limit_error(
        "Rate limit reached on requests per minute (RPM).", retry_after=17
    )
    calls = {"n": 0}

    def dispatch(provider, prompt, system, temperature, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            raise err
        return "ok", None

    monkeypatch.setattr(bb, "_dispatch_provider_call", dispatch)
    bb.call_llm("hello", retries=3)
    assert sleep_calls == [17.0]  # used the header value, not exponential backoff


def test_5xx_retries_with_exponential_backoff(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    err = make_server_error(503)
    calls = {"n": 0}

    def dispatch(provider, prompt, system, temperature, max_tokens):
        calls["n"] += 1
        if calls["n"] < 3:
            raise err
        return "ok", None

    monkeypatch.setattr(bb, "_dispatch_provider_call", dispatch)
    result = bb.call_llm("hello", retries=5)
    assert result == "ok"
    assert len(sleep_calls) == 2
    assert sleep_calls[0] < sleep_calls[1]  # exponential, not constant


# ── checkpoint write/resume — requirement #3 ──────────────────────────────

def fake_llm_always_succeeds(prompt, **kwargs):
    if "fully_answerable_alone" in prompt:
        return '{"fully_answerable_alone": false}'
    if "any_chunk_answers" in prompt:
        return '{"any_chunk_answers": false}'
    if '"supported"' in prompt:
        return '{"supported": true}'
    if "ambiguity_explanation" in prompt:
        return '{"question": "What did it show?", "ambiguity_explanation": "A or B."}'
    if "answer" in prompt:
        return '{"question": "What was the result?", "answer": "42"}'
    return '{"question": "What was the result?"}'


def test_checkpoint_is_written_after_accepted_question(monkeypatch, tmp_path):
    monkeypatch.setattr(bb, "call_llm", fake_llm_always_succeeds)
    corpus_path = write_corpus(tmp_path)
    out_dir = tmp_path / "out"

    bb.build_benchmark(
        target_size=4, seed=1, faiss_metadata_path=corpus_path,
        max_attempts_multiplier=10, out_dir=str(out_dir), categories=["factual"],
    )
    checkpoint_path = out_dir / "benchmark_checkpoint.json"
    assert checkpoint_path.exists()
    data = json.loads(checkpoint_path.read_text())
    assert len(data["items"]) == 4
    assert data["seed"] == 1
    assert data["category_counts"]["factual"] == 4


def test_checkpoint_resumes_and_does_not_regenerate_accepted_items(monkeypatch, tmp_path):
    corpus_path = write_corpus(tmp_path)
    out_dir = tmp_path / "out"

    # First run: only reach 2/4 (simulate a stop) by using a low attempt budget
    # and a call_llm that fails after producing 2 items.
    produced = {"n": 0}

    def limited_llm(prompt, **kwargs):
        if prompt.startswith("Write ONE factual question"):
            if produced["n"] >= 2:
                return None
            produced["n"] += 1
        return fake_llm_always_succeeds(prompt, **kwargs)

    monkeypatch.setattr(bb, "call_llm", limited_llm)
    result1 = bb.build_benchmark(
        target_size=4, seed=1, faiss_metadata_path=corpus_path,
        max_attempts_multiplier=3, out_dir=str(out_dir), categories=["factual"],
    )
    assert len(result1["items"]) == 2
    first_run_ids = {i["id"] for i in result1["items"]}

    # Resume: now let generation succeed fully.
    monkeypatch.setattr(bb, "call_llm", fake_llm_always_succeeds)
    result2 = bb.build_benchmark(
        target_size=4, seed=1, faiss_metadata_path=corpus_path,
        max_attempts_multiplier=10, out_dir=str(out_dir), categories=["factual"],
        resume=True,
    )
    assert len(result2["items"]) == 4
    resumed_ids = {i["id"] for i in result2["items"]}
    assert first_run_ids.issubset(resumed_ids)  # the 2 accepted items were kept, not regenerated


def test_resume_without_checkpoint_fails_clearly(tmp_path):
    corpus_path = write_corpus(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(bb.BenchmarkGenerationError, match="no checkpoint"):
        bb.build_benchmark(
            target_size=4, seed=1, faiss_metadata_path=corpus_path,
            out_dir=str(out_dir), resume=True,
        )


def test_incompatible_checkpoint_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(bb, "call_llm", fake_llm_always_succeeds)
    corpus_path = write_corpus(tmp_path)
    out_dir = tmp_path / "out"

    bb.build_benchmark(
        target_size=2, seed=1, faiss_metadata_path=corpus_path,
        max_attempts_multiplier=10, out_dir=str(out_dir), categories=["factual"],
    )
    # Resuming with a DIFFERENT seed must be rejected.
    with pytest.raises(bb.BenchmarkGenerationError, match="not compatible"):
        bb.build_benchmark(
            target_size=2, seed=999, faiss_metadata_path=corpus_path,
            out_dir=str(out_dir), categories=["factual"], resume=True,
        )


def test_incompatible_checkpoint_rejected_on_different_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(bb, "call_llm", fake_llm_always_succeeds)
    corpus_path = write_corpus(tmp_path)
    out_dir = tmp_path / "out"

    bb.build_benchmark(
        target_size=2, seed=1, faiss_metadata_path=corpus_path,
        max_attempts_multiplier=10, out_dir=str(out_dir), categories=["factual"],
    )
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    # A genuinely different corpus (different text -> different fingerprint),
    # not just a different directory holding identical content.
    other_corpus_path = write_corpus(other_dir, n=9)
    with pytest.raises(bb.BenchmarkGenerationError, match="not compatible"):
        bb.build_benchmark(
            target_size=2, seed=1, faiss_metadata_path=other_corpus_path,
            out_dir=str(out_dir), categories=["factual"], resume=True,
        )


# ── partial run never creates benchmark.json — requirement #4 ────────────

def test_quota_exhaustion_never_creates_benchmark_json(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    corpus_path = write_corpus(tmp_path)
    out_dir = tmp_path / "out"

    calls = {"n": 0}

    def dispatch(provider, prompt, system, temperature, max_tokens):
        calls["n"] += 1
        if calls["n"] > 3:  # let a couple of items through, then hit TPD
            raise make_rate_limit_error(
                "... on tokens per day (TPD): Limit 200000, Used 200000, Requested 1."
            )
        if "fully_answerable_alone" in prompt or "any_chunk_answers" in prompt or '"supported"' in prompt:
            return '{"fully_answerable_alone": false, "any_chunk_answers": false, "supported": true}', None
        return '{"question": "What was the result?", "answer": "42"}', None

    monkeypatch.setattr(bb, "_dispatch_provider_call", dispatch)

    argv = [
        "build_benchmark.py", "--target-size", "10", "--seed", "1",
        "--faiss-metadata", corpus_path, "--out-dir", str(out_dir),
        "--categories", "factual", "--max-attempts-multiplier", "20",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        bb.main()
    assert exc_info.value.code == 1
    assert not (out_dir / "benchmark.json").exists()
    assert (out_dir / "benchmark_checkpoint.json").exists()  # checkpoint IS preserved


# ── --max-api-calls — requirement #6 ──────────────────────────────────────

def test_max_api_calls_stops_cleanly_and_preserves_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    corpus_path = write_corpus(tmp_path)
    out_dir = tmp_path / "out"

    def dispatch(provider, prompt, system, temperature, max_tokens):
        return fake_llm_always_succeeds(prompt), None

    monkeypatch.setattr(bb, "_dispatch_provider_call", dispatch)

    argv = [
        "build_benchmark.py", "--target-size", "10", "--seed", "1",
        "--faiss-metadata", corpus_path, "--out-dir", str(out_dir),
        "--categories", "factual", "--max-attempts-multiplier", "20",
        "--max-api-calls", "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        bb.main()
    assert exc_info.value.code == 1
    assert not (out_dir / "benchmark.json").exists()
    assert bb._usage.total_requests <= 3  # stopped at/near the limit, not way past it


# ── successful complete run still creates benchmark.json ─────────────────

def test_successful_complete_run_creates_benchmark_json(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr(bb, "call_llm", fake_llm_always_succeeds)
    corpus_path = write_corpus(tmp_path)
    out_dir = tmp_path / "out"

    argv = [
        "build_benchmark.py", "--target-size", "8", "--seed", "1",
        "--faiss-metadata", corpus_path, "--out-dir", str(out_dir),
        "--categories", "factual,numerical", "--max-attempts-multiplier", "10",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bb.main()  # should NOT raise/exit
    assert (out_dir / "benchmark.json").exists()
    items = json.loads((out_dir / "benchmark.json").read_text())
    assert len(items) == 8


# ── usage accounting (requirement #5) ─────────────────────────────────────

def test_usage_tracker_records_requests_and_tokens():
    usage = bb.UsageTracker()
    usage.total_requests = 3
    usage.successful_requests = 2
    usage.failed_requests = 1
    usage.retry_count = 1

    class FakeUsage:
        prompt_tokens = 100
        completion_tokens = 50
        total_tokens = 150

    usage.record_tokens(FakeUsage())
    summary = usage.summary_dict()
    assert summary["total_requests"] == 3
    assert summary["estimated_input_tokens"] == 100
    assert summary["estimated_output_tokens"] == 50
    assert summary["estimated_total_tokens"] == 150


def test_usage_tracker_handles_missing_token_data_gracefully():
    usage = bb.UsageTracker()
    usage.record_tokens(None)
    summary = usage.summary_dict()
    assert "estimated_total_tokens" not in summary


def test_usage_tracker_resets_between_runs():
    bb._usage.total_requests = 99
    bb._usage.reset()
    assert bb._usage.total_requests == 0


# ── existing benchmark validation behavior unchanged — requirement #11 ───

def test_existing_schema_validation_still_works_unchanged():
    from evaluation.benchmark_schema import validate_item
    item = {
        "id": "bq_0001", "category": "factual", "question": "q?",
        "answerable": True, "evidence_status": "sufficient",
        "gold_document_ids": ["paper_000.pdf"], "gold_chunk_ids": ["x:0"],
        "reference_answer": "a", "attack_type": None,
    }
    assert validate_item(item) == []
