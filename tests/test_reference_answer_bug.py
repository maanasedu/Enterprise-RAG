"""Regression tests for the reference_answer acceptance-gate bug:

    if (... and item.get("reference_answer") and ...)

used to GATE whether validation ran at all, so an item with a missing or
empty reference_answer skipped validation entirely instead of being
rejected by it, and reached items.append() unchecked (real-world symptom:
bq_numerical_98f5f3c89847 reached a 250-item checkpoint with an empty
reference_answer, caught only by the final pre-write schema pass).

Also covers the checkpoint-repair path that makes --resume safe against a
checkpoint that already contains such an invalid item from before this fix.

All LLM calls are mocked; no network calls.
"""
import importlib.util
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "build_benchmark", os.path.join(_ROOT, "scripts", "build_benchmark.py")
)
bb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bb)

from evaluation.benchmark_schema import validate_item  # noqa: E402


def write_corpus(tmp_path, n=6):
    corpus = [
        {"text": f"Paper {i} discusses topic {i} with result {i * 10}.",
         "metadata": {"filename": f"paper_{i:03d}.pdf", "chunk_index": 0}}
        for i in range(n)
    ]
    path = tmp_path / "faiss_metadata.json"
    path.write_text(json.dumps(corpus))
    return str(path)


def ok_llm(prompt, **kw):
    if '"supported"' in prompt:
        return '{"supported": true}'
    return '{"question": "What was the result?", "answer": "42"}'


# ── 1/2/3: missing / empty / valid reference_answer at acceptance ────────

def test_answerable_item_with_missing_reference_answer_is_rejected(monkeypatch, tmp_path):
    """The exact shape of the real bug, tested directly against the
    ACCEPTANCE LOOP (not just the generator), so the fix is proven at the
    point the bug actually lived."""
    monkeypatch.setattr(bb, "call_llm", ok_llm)
    corpus = write_corpus(tmp_path)

    def bad_gen(chunk, category, rng):
        return {
            "category": category, "question": "What was the result?",
            "answerable": True, "evidence_status": "sufficient",
            "reference_answer": None, "attack_type": None,
            "_gold_chunks": [chunk], "_seed_chunk_ids": [bb.chunk_identity_key(chunk)],
        }
    monkeypatch.setattr(bb, "gen_single_chunk_question", bad_gen)

    result = bb.build_benchmark(
        target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"],
        max_attempts_multiplier=2, out_dir=str(tmp_path / "out"),
    )
    assert result["items"] == []
    # max_attempts_multiplier=2 * target_size=2 = 4 attempts, every one
    # rejected for the same reason.
    assert result["counters"]["rejected_by_reason"]["missing_reference_answer"] == 4


def test_answerable_item_with_empty_string_reference_answer_is_rejected(monkeypatch, tmp_path):
    """The literal real-world case: LLM returns {"answer": ""} — a present
    key, an empty string. `item.get("reference_answer")` on "" is falsy,
    which is exactly what let this slip through the old gate."""
    monkeypatch.setattr(bb, "call_llm", lambda p, **k: '{"question": "Q?", "answer": ""}')
    corpus = write_corpus(tmp_path)
    result = bb.build_benchmark(
        target_size=1, seed=1, faiss_metadata_path=corpus, categories=["factual"],
        max_attempts_multiplier=2, out_dir=str(tmp_path / "out"),
    )
    assert result["items"] == []
    assert result["counters"]["rejected_by_reason"]["missing_reference_answer"] == 2
    # No judge call should even have been spent validating an empty answer.
    assert result["counters"]["reference_answer_validation_passed"] == 0


def test_answerable_item_with_whitespace_only_reference_answer_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(bb, "call_llm", lambda p, **k: '{"question": "Q?", "answer": "   "}')
    corpus = write_corpus(tmp_path)
    result = bb.build_benchmark(
        target_size=1, seed=1, faiss_metadata_path=corpus, categories=["factual"],
        max_attempts_multiplier=2, out_dir=str(tmp_path / "out"),
    )
    assert result["items"] == []
    assert result["counters"]["rejected_by_reason"]["missing_reference_answer"] == 2


def test_valid_answerable_item_with_reference_answer_is_still_accepted(monkeypatch, tmp_path):
    """The fix must not be over-strict: a normal, valid item still passes."""
    monkeypatch.setattr(bb, "call_llm", ok_llm)
    corpus = write_corpus(tmp_path)
    result = bb.build_benchmark(
        target_size=2, seed=1, faiss_metadata_path=corpus, categories=["factual"],
        out_dir=str(tmp_path / "out"),
    )
    assert len(result["items"]) == 2
    for item in result["items"]:
        assert item["reference_answer"] == "42"


def test_ambiguous_item_with_missing_ambiguity_explanation_is_rejected(monkeypatch, tmp_path):
    """Same bug class applies to 'ambiguous' items: gen_ambiguous_question
    sets reference_answer to the explanation text, which the LLM can also
    return empty/missing — must be caught too, not just for 'factual'."""
    monkeypatch.setattr(bb, "call_llm", lambda p, **k: '{"question": "What did it show?"}')
    corpus = write_corpus(tmp_path)
    result = bb.build_benchmark(
        target_size=1, seed=1, faiss_metadata_path=corpus, categories=["ambiguous"],
        max_attempts_multiplier=2, out_dir=str(tmp_path / "out"),
    )
    assert result["items"] == []
    assert result["counters"]["rejected_by_reason"]["missing_reference_answer"] == 2


# ── 4: checkpoint never gets an invalid item appended ─────────────────────

def test_checkpoint_never_contains_an_item_with_empty_reference_answer(monkeypatch, tmp_path):
    calls = {"n": 0}

    def flaky_llm(prompt, **kw):
        if '"supported"' in prompt:
            return '{"supported": true}'
        calls["n"] += 1
        # Every third generated answer comes back empty — simulates the
        # real intermittent Groq behavior that produced the bug.
        if calls["n"] % 3 == 0:
            return '{"question": "Q?", "answer": ""}'
        return '{"question": "Q?", "answer": "42"}'

    monkeypatch.setattr(bb, "call_llm", flaky_llm)
    corpus = write_corpus(tmp_path)
    out = tmp_path / "out"
    result = bb.build_benchmark(
        target_size=6, seed=1, faiss_metadata_path=corpus, categories=["factual"],
        max_attempts_multiplier=5, out_dir=str(out),
    )
    assert len(result["items"]) == 6
    for item in result["items"]:
        assert item["reference_answer"] and item["reference_answer"].strip()

    checkpoint = json.loads((out / "benchmark_checkpoint.json").read_text())
    for item in checkpoint["items"]:
        assert item["reference_answer"] and item["reference_answer"].strip()
    assert result["counters"]["rejected_by_reason"]["missing_reference_answer"] >= 1


# ── 9: checkpoint-repair behavior ─────────────────────────────────────────

def _inject_invalid_item(checkpoint_path, corpus):
    """Simulates the real scenario: a checkpoint written by the OLD buggy
    code contains one item with an empty reference_answer among otherwise
    valid items."""
    checkpoint = json.loads(open(checkpoint_path).read())
    bad = dict(checkpoint["items"][0])
    bad["id"] = "bq_numerical_deadbeef0001"
    bad["category"] = "numerical"
    bad["reference_answer"] = ""  # the actual real-world defect
    checkpoint["items"].append(bad)
    checkpoint["used_ids"] = checkpoint.get("used_ids", []) + [bad["id"]]
    checkpoint["category_counts"]["numerical"] = checkpoint["category_counts"].get("numerical", 0) + 1
    checkpoint["counters"]["total_accepted"] += 1
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)


def test_repair_checkpoint_items_drops_only_invalid_items():
    valid = {"id": "bq_a", "category": "factual", "question": "q", "answerable": True,
             "evidence_status": "sufficient", "gold_document_ids": ["paper_000.pdf"],
             "gold_chunk_ids": ["text-hash:aaa"], "reference_answer": "ans", "attack_type": None}
    invalid = {"id": "bq_b", "category": "numerical", "question": "q2", "answerable": True,
               "evidence_status": "sufficient", "gold_document_ids": ["paper_000.pdf"],
               "gold_chunk_ids": ["text-hash:bbb"], "reference_answer": "", "attack_type": None}
    kept, removed = bb._repair_checkpoint_items(
        [valid, invalid],
        valid_chunk_ids={"text-hash:aaa", "text-hash:bbb"},
        valid_document_ids={"paper_000.pdf"},
        chunk_id_to_document_id={"text-hash:aaa": "paper_000.pdf", "text-hash:bbb": "paper_000.pdf"},
    )
    assert [i["id"] for i in kept] == ["bq_a"]
    assert len(removed) == 1 and removed[0]["id"] == "bq_b"
    assert any("reference_answer" in e for e in removed[0]["_repair_errors"])


def test_repair_checkpoint_items_is_noop_when_everything_valid():
    valid = {"id": "bq_a", "category": "factual", "question": "q", "answerable": True,
             "evidence_status": "sufficient", "gold_document_ids": ["paper_000.pdf"],
             "gold_chunk_ids": ["text-hash:aaa"], "reference_answer": "ans", "attack_type": None}
    kept, removed = bb._repair_checkpoint_items(
        [valid], valid_chunk_ids={"text-hash:aaa"}, valid_document_ids={"paper_000.pdf"},
        chunk_id_to_document_id={"text-hash:aaa": "paper_000.pdf"},
    )
    assert kept == [valid] and removed == []


def test_resume_repairs_a_checkpoint_with_a_pre_existing_invalid_item(monkeypatch, tmp_path):
    """End-to-end: build a small valid checkpoint, hand-inject an invalid
    item the way the OLD bug would have produced, then resume — the
    already-valid items must be preserved untouched, the invalid item
    must be dropped, and its category slot regenerated to reach the
    target again."""
    corpus = write_corpus(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(bb, "call_llm", ok_llm)

    first = bb.build_benchmark(
        target_size=4, seed=1, faiss_metadata_path=corpus,
        categories=["factual", "numerical"], out_dir=str(out),
    )
    assert len(first["items"]) == 4
    original_ids = {i["id"] for i in first["items"]}

    _inject_invalid_item(str(out / "benchmark_checkpoint.json"), corpus)

    second = bb.build_benchmark(
        target_size=4, seed=1, faiss_metadata_path=corpus,
        categories=["factual", "numerical"], out_dir=str(out), resume=True,
    )
    assert len(second["items"]) == 4  # back to 4, not 5
    for item in second["items"]:
        assert item["reference_answer"] and item["reference_answer"].strip()
    resumed_ids = {i["id"] for i in second["items"]}
    assert original_ids <= resumed_ids  # the 4 originally-valid items were kept, not regenerated
    assert "bq_numerical_deadbeef0001" not in resumed_ids
    assert second["counters"]["rejected_by_reason"]["checkpoint_repair_invalid_item"] == 1


def test_repair_writes_corrected_checkpoint_to_disk_immediately(monkeypatch, tmp_path):
    """Even if the resumed run is then immediately interrupted, the repair
    itself must already be persisted — the on-disk checkpoint must never
    still hold the invalid item after a --resume ran. Uses a checkpoint
    where the invalid item's removal drops the count below target, so a
    (mocked, immediately-failing) generation call is actually required —
    proving the repair was saved even though nothing else got a chance to
    save afterward."""
    corpus = write_corpus(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(bb, "call_llm", ok_llm)
    bb.build_benchmark(target_size=1, seed=1, faiss_metadata_path=corpus,
                       categories=["factual"], out_dir=str(out))

    checkpoint_path = out / "benchmark_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    bad = dict(checkpoint["items"][0])
    bad["id"] = "bq_factual_deadbeef0002"
    bad["reference_answer"] = ""  # the real-world defect
    checkpoint["items"].append(bad)
    checkpoint["target_size"] = 2  # pretend the original (buggy) run targeted 2
    checkpoint["used_ids"].append(bad["id"])
    checkpoint_path.write_text(json.dumps(checkpoint))

    def dies_immediately(*a, **k):
        raise bb.BenchmarkQuotaExhaustedError("tokens per day")
    monkeypatch.setattr(bb, "call_llm", dies_immediately)
    with pytest.raises(bb.BenchmarkQuotaExhaustedError):
        bb.build_benchmark(target_size=2, seed=1, faiss_metadata_path=corpus,
                           categories=["factual"], out_dir=str(out), resume=True)

    on_disk = json.loads(checkpoint_path.read_text())
    assert "bq_factual_deadbeef0002" not in {i["id"] for i in on_disk["items"]}
    assert len(on_disk["items"]) == 1  # the repair (drop to 1) was saved even
                                        # though the very next step then failed


def test_repaired_benchmark_passes_full_final_validation(monkeypatch, tmp_path):
    """The scenario from the report: repair a checkpoint, let generation
    finish, and confirm the final validate_benchmark pass (the same one
    main() runs before writing benchmark.json) is clean."""
    from evaluation.benchmark_schema import validate_benchmark
    corpus = write_corpus(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(bb, "call_llm", ok_llm)
    bb.build_benchmark(target_size=3, seed=1, faiss_metadata_path=corpus,
                       categories=["factual"], out_dir=str(out))
    _inject_invalid_item(str(out / "benchmark_checkpoint.json"), corpus)

    result = bb.build_benchmark(
        target_size=3, seed=1, faiss_metadata_path=corpus,
        categories=["factual"], out_dir=str(out), resume=True,
    )
    assert len(result["items"]) == 3
    validation = validate_benchmark(
        result["items"], valid_chunk_ids=result["all_chunk_ids"],
        valid_document_ids=result["all_document_ids"],
        chunk_id_to_document_id=result["chunk_id_to_document_id"],
    )
    assert not validation["errors_by_id"] and not validation["duplicate_ids"]
    assert not result["shortfall"]
