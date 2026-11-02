"""Tests for the reliability patch: robust judge parsing, reasoning-model
token floor, distinct judge-error rejection reasons, initial checkpoint,
per-attempt checkpointing, rng-state resume, honest termination messages.
All LLM calls are mocked; no network."""
import importlib.util
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location("build_benchmark", os.path.join(_ROOT, "scripts", "build_benchmark.py"))
bb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bb)


def write_corpus(tmp_path, n=6):
    corpus = [{"text": f"Paper {i} discusses topic {i} with result {i * 10}.",
               "metadata": {"filename": f"paper_{i:03d}.pdf", "chunk_index": 0}} for i in range(n)]
    path = tmp_path / "faiss_metadata.json"
    path.write_text(json.dumps(corpus))
    return str(path)


def ok_llm(prompt, **kw):
    if '"supported"' in prompt:
        return '{"supported": true}'
    return '{"question": "What was the result?", "answer": "42"}'


# ── judge parsing ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ('{"supported": true}', True),
    ('{"supported": false}', False),
    ('```json\n{"supported": false}\n```', False),
    ('Sure! {"supported": true} hope that helps', True),
    ('{"supported": "false"}', False),      # string "false" must NOT become True
    ('{"supported": "true"}', True),
    ('supported: true', True),
    ('', None), (None, None), ('{"supported": "maybe"}', None), ('no verdict here', None),
])
def test_parse_bool_field(raw, expected):
    assert bb._parse_bool_field(raw, "supported") is expected


def test_judge_reasks_once_on_unparseable_then_succeeds(monkeypatch):
    replies = iter(["garbage", '{"supported": true}'])
    monkeypatch.setattr(bb, "call_llm", lambda *a, **k: next(replies))
    assert bb._judge_bool("p", "supported") is True


def test_judge_returns_none_when_never_parseable(monkeypatch):
    monkeypatch.setattr(bb, "call_llm", lambda *a, **k: None)
    assert bb._judge_bool("p", "supported") is None


def test_reference_answer_string_false_is_not_accepted(monkeypatch):
    monkeypatch.setattr(bb, "call_llm", lambda *a, **k: '{"supported": "false"}')
    assert bb.reference_answer_is_supported("q", "evidence", "a") is False


# ── reasoning-model root cause ───────────────────────────────────────────

def test_groq_call_uses_token_floor_for_small_judge_budget(monkeypatch):
    import groq
    seen = {}

    class C:
        def __init__(self, api_key=None):
            self.chat = type("Ch", (), {"completions": self})()
        def create(self, **kw):
            seen.update(kw)
            return type("R", (), {"choices": [type("X", (), {"message": type("M", (), {"content": "ok"})()})()]})()

    monkeypatch.setattr(groq, "Groq", C)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    content, _ = bb._dispatch_provider_call("groq", "p", None, 0.0, 50)
    assert seen["max_tokens"] >= 1500 and seen["reasoning_effort"] == "low"


def test_empty_llm_response_is_failed_request_not_success(monkeypatch):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setattr(bb.time, "sleep", lambda s: None)
    monkeypatch.setattr(bb, "_dispatch_provider_call", lambda *a, **k: ("", None))
    bb._usage.reset()
    assert bb.call_llm("p", retries=2) is None
    assert bb._usage.successful_requests == 0 and bb._usage.failed_requests == 2


def test_judge_error_and_unsupported_have_distinct_rejection_reasons(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path)

    def llm(prompt, **kw):
        if '"supported"' in prompt:
            return None  # judge never answers
        return '{"question": "q?", "answer": "a"}'

    monkeypatch.setattr(bb, "call_llm", llm)
    r = bb.build_benchmark(target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"],
                           max_attempts_multiplier=2, out_dir=str(tmp_path / "o"))
    reasons = r["counters"]["rejected_by_reason"]
    assert reasons["reference_answer_judge_error"] == 4 and not r["items"]
    assert "reference_answer_unsupported" not in reasons

    monkeypatch.setattr(bb, "call_llm", lambda p, **k: '{"supported": false}' if '"supported"' in p else '{"question": "q?", "answer": "a"}')
    r = bb.build_benchmark(target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"],
                           max_attempts_multiplier=1, out_dir=str(tmp_path / "o2"))
    assert r["counters"]["rejected_by_reason"]["reference_answer_unsupported"] == 2
    assert not r["items"]  # unsupported answers are never accepted


def test_validation_passes_end_to_end_with_working_judge(monkeypatch, tmp_path):
    monkeypatch.setattr(bb, "call_llm", ok_llm)
    r = bb.build_benchmark(target_size=3, seed=1, faiss_metadata_path=write_corpus(tmp_path),
                           categories=["factual"], out_dir=str(tmp_path / "o"))
    assert len(r["items"]) == 3 and not r["validation"]["errors_by_id"]


# ── checkpointing ────────────────────────────────────────────────────────

def test_initial_checkpoint_exists_when_stopped_before_first_item(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    monkeypatch.setattr(bb, "_dispatch_provider_call", lambda *a, **k: (ok_llm(a[1]), None))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["b", "--target-size", "5", "--categories", "factual", "--seed", "42",
                                      "--max-api-calls", "0", "--faiss-metadata", write_corpus(tmp_path),
                                      "--out-dir", str(out)])
    with pytest.raises(SystemExit) as e:
        bb.main()
    assert e.value.code == 1
    cp = json.loads((out / "benchmark_checkpoint.json").read_text())
    assert cp["items"] == [] and cp["used_ids"] == [] and cp["seed"] == 42
    assert cp["plan"] == {"factual": 5} and cp["corpus_fingerprint"] and cp["checkpoint_version"] == 2
    assert not (out / "benchmark.json").exists()


def test_quota_exhaustion_before_first_item_leaves_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")

    def boom(*a, **k):
        raise bb.BenchmarkQuotaExhaustedError("tokens per day")
    monkeypatch.setattr(bb, "_dispatch_provider_call", boom)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["b", "--target-size", "2", "--categories", "factual",
                                      "--faiss-metadata", write_corpus(tmp_path), "--out-dir", str(out)])
    with pytest.raises(SystemExit):
        bb.main()
    assert json.loads((out / "benchmark_checkpoint.json").read_text())["items"] == []


def test_resume_from_zero_item_checkpoint(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path)
    out = str(tmp_path / "o")
    # zero-item checkpoint via an immediate --max-api-calls limit (real call_llm, mocked dispatch)
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "groq")
    monkeypatch.setattr(bb, "_dispatch_provider_call", lambda *a, **k: (ok_llm(a[1]), None))
    with pytest.raises(bb.BenchmarkAPICallLimitReached):
        bb.build_benchmark(target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"],
                           out_dir=out, max_api_calls=0)
    assert json.loads(open(os.path.join(out, "benchmark_checkpoint.json")).read())["items"] == []
    r = bb.build_benchmark(target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"],
                           out_dir=out, resume=True)
    assert len(r["items"]) == 2


def test_rejected_attempts_are_checkpointed_and_survive_resume(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path)
    out = str(tmp_path / "o")
    state = {"gen": 0}

    def llm(prompt, **kw):
        if '"supported"' in prompt:
            return '{"supported": false}'
        state["gen"] += 1
        return '{"question": "q%d?", "answer": "a"}' % state["gen"]

    monkeypatch.setattr(bb, "call_llm", llm)
    bb.build_benchmark(target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"],
                       out_dir=out, max_attempts_multiplier=2)
    cp = json.loads(open(os.path.join(out, "benchmark_checkpoint.json")).read())
    assert cp["counters"]["total_attempts"] == 4 and cp["counters"]["total_rejected"] == 4
    assert cp["counters"]["rejected_by_reason"] == {"reference_answer_unsupported": 4}
    assert cp["attempts_by_category"] == {"factual": 4} and cp["rng_state"]

    monkeypatch.setattr(bb, "call_llm", ok_llm)
    r = bb.build_benchmark(target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"],
                           out_dir=out, resume=True, max_attempts_multiplier=5)
    assert r["counters"]["total_rejected"] == 4  # not reset
    assert r["counters"]["rejected_by_reason"]["reference_answer_unsupported"] == 4
    assert len(r["items"]) == 2


def test_incomplete_attempt_not_counted_as_rejected(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path)
    out = str(tmp_path / "o")
    monkeypatch.setattr(bb, "call_llm", ok_llm)
    bb.build_benchmark(target_size=1, seed=1, faiss_metadata_path=corpus, categories=["factual"], out_dir=out)

    calls = {"n": 0}

    def dying(prompt, **kw):
        calls["n"] += 1
        raise bb.BenchmarkQuotaExhaustedError("tokens per day")
    monkeypatch.setattr(bb, "call_llm", dying)
    with pytest.raises(bb.BenchmarkQuotaExhaustedError):
        bb.build_benchmark(target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"],
                           out_dir=out + "2")
    cp = json.loads(open(os.path.join(out + "2", "benchmark_checkpoint.json")).read())
    assert cp["counters"]["total_rejected"] == 0 and cp["counters"]["total_attempts"] == 0


def test_resume_does_not_duplicate_accepted_items(monkeypatch, tmp_path):
    corpus = write_corpus(tmp_path)
    out = str(tmp_path / "o")
    monkeypatch.setattr(bb, "call_llm", ok_llm)
    bb.build_benchmark(target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"], out_dir=out)
    r = bb.build_benchmark(target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"],
                           out_dir=out, resume=True)
    ids = [i["id"] for i in r["items"]]
    assert len(ids) == len(set(ids)) == 2


# ── honest termination message ───────────────────────────────────────────

def test_status_report_does_not_claim_missing_checkpoint(tmp_path, capsys):
    bb._report_checkpoint_status(str(tmp_path / "nope.json"), 5)
    out = capsys.readouterr().out
    assert "No checkpoint was written" in out and "Checkpoint preserved" not in out


def test_status_report_prints_path_when_checkpoint_exists(tmp_path, capsys):
    p = tmp_path / "cp.json"
    p.write_text(json.dumps({"items": [], "counters": {"rejected_by_reason": {"x": 2}}}))
    bb._report_checkpoint_status(str(p), 5)
    out = capsys.readouterr().out
    assert "Checkpoint preserved at" in out and "0/5" in out and "x=2" in out
