"""Schema definition and validation for the Dimension 1 fixed evaluation
benchmark (data/evaluation/benchmark.json).

This module has NO external dependencies (no API calls, no network) so it
can be imported and unit tested in any environment, including one with no
LLM API keys configured.

Schema (one JSON object per benchmark item):

    {
      "id": str,                      # unique, stable across regenerations
      "category": str,                # one of CATEGORIES
      "question": str,
      "answerable": bool,
      "evidence_status": str,         # one of EVIDENCE_STATUSES
      "gold_document_ids": [str, ...],
      "gold_chunk_ids": [str, ...],
      "reference_answer": str | None,
      "attack_type": str | None,      # required (non-null) iff category == "adversarial"
      "injected_fixture": dict | absent,  # required iff attack_type == "indirect_document_injection"
    }

`injected_fixture` (only present for attack_type == "indirect_document_injection")
is a synthetic, benchmark-only fixture representing a poisoned document
chunk — it is NEVER written into the production corpus/index. Required
sub-fields: fixture_id, source_question_chunk_id (the REAL corpus chunk
id the benign `question` is actually answerable from), injected_text (the
malicious instruction text a future security-eval harness would insert
into retrieved context alongside the real gold evidence), and
injection_note (a human-readable description of how it's meant to be
used). See docs/benchmark.md #2.

See docs/benchmark.md for the full field-by-field rationale.
"""
from typing import Dict, Iterable, List, Optional, Sequence

CATEGORIES = [
    "factual",
    "numerical",
    "multi_document",
    "comparison",
    "conversational",
    "ambiguous",
    "unanswerable",
    "adversarial",
]

EVIDENCE_STATUSES = ["sufficient", "insufficient", "ambiguous"]

# Open set (informational only) — validation just requires attack_type to be
# a non-empty string for adversarial items, not one of these exact values,
# since new attack patterns may need to be added later.
KNOWN_ATTACK_TYPES = [
    "direct_prompt_injection",
    "indirect_document_injection",
    "instruction_override",
    "embedded_malicious_instruction",
]

REQUIRED_FIELDS = [
    "id",
    "category",
    "question",
    "answerable",
    "evidence_status",
    "gold_document_ids",
    "gold_chunk_ids",
    "reference_answer",
    "attack_type",
]

# Expected system behavior per evidence_status, for documentation /
# programmatic access by future Adaptive RAG / refusal-evaluation code.
EVIDENCE_STATUS_EXPECTED_BEHAVIOR = {
    "sufficient": (
        "System should retrieve the gold evidence and answer using it. "
        "A correct system answers; refusing is a false-refusal error."
    ),
    "insufficient": (
        "Corpus does not contain the answer. System should refuse / say it "
        "cannot answer from the available documents. Answering with a "
        "specific claim is a hallucination error."
    ),
    "ambiguous": (
        "Question admits multiple plausible interpretations or the "
        "evidence does not resolve a single confident answer. Expected "
        "future behavior (Adaptive RAG, not yet implemented): ask a "
        "clarifying question or explicitly surface the alternative "
        "interpretations rather than silently picking one and answering "
        "as if it were unambiguous."
    ),
}


def validate_item(
    item: Dict,
    valid_chunk_ids: Optional[set] = None,
    valid_document_ids: Optional[set] = None,
    chunk_id_to_document_id: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Validate one benchmark item. Returns a list of error strings (empty
    list means valid). Never raises — callers decide what to do with a
    non-empty error list (reject the item, log it, fail a test, etc.).

    `valid_chunk_ids` / `valid_document_ids`: if provided, gold ids are
    checked for existence against the actual corpus.
    `chunk_id_to_document_id`: if provided, gold chunks are checked to
    actually belong to one of the item's gold_document_ids.
    """
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in item:
            errors.append(f"missing required field: {field}")
    if errors:
        # Can't safely check further without the required fields present.
        return errors

    if not isinstance(item["id"], str) or not item["id"]:
        errors.append("id must be a non-empty string")

    if item["category"] not in CATEGORIES:
        errors.append(f"invalid category: {item['category']!r}")

    if not isinstance(item["question"], str) or not item["question"].strip():
        errors.append("question must be a non-empty string")

    if not isinstance(item["answerable"], bool):
        errors.append("answerable must be a bool")

    if item["evidence_status"] not in EVIDENCE_STATUSES:
        errors.append(f"invalid evidence_status: {item['evidence_status']!r}")

    if not isinstance(item["gold_document_ids"], list) or not all(
        isinstance(x, str) for x in item["gold_document_ids"]
    ):
        errors.append("gold_document_ids must be a list of strings")

    if not isinstance(item["gold_chunk_ids"], list) or not all(
        isinstance(x, str) for x in item["gold_chunk_ids"]
    ):
        errors.append("gold_chunk_ids must be a list of strings")

    # answerability / evidence consistency
    if item.get("answerable") is False:
        if item.get("evidence_status") != "insufficient":
            errors.append(
                "answerable=False items must have evidence_status='insufficient'"
            )
        if item.get("gold_document_ids"):
            errors.append("unanswerable items must have empty gold_document_ids")
        if item.get("gold_chunk_ids"):
            errors.append("unanswerable items must have empty gold_chunk_ids")
        if item.get("reference_answer") is not None:
            errors.append("unanswerable items must have reference_answer=null")
    else:
        if item.get("evidence_status") == "insufficient":
            errors.append(
                "answerable=True items must not have evidence_status='insufficient'"
            )
        if item.get("category") != "ambiguous" and not item.get("gold_chunk_ids"):
            errors.append(
                "answerable items (other than 'ambiguous') must have at least one gold_chunk_id"
            )
        if not item.get("gold_document_ids"):
            errors.append("answerable items must have at least one gold_document_id")
        if not isinstance(item.get("reference_answer"), str) or not item[
            "reference_answer"
        ].strip():
            errors.append("answerable items must have a non-empty reference_answer")

    # adversarial-specific
    if item["category"] == "adversarial":
        if not isinstance(item.get("attack_type"), str) or not item["attack_type"].strip():
            errors.append("adversarial items must have a non-null attack_type")
        if item.get("attack_type") == "indirect_document_injection":
            fixture = item.get("injected_fixture")
            if not isinstance(fixture, dict):
                errors.append(
                    "indirect_document_injection items must have an "
                    "injected_fixture object (see docs/benchmark.md #2)"
                )
            else:
                for key in ("fixture_id", "source_question_chunk_id", "injected_text", "injection_note"):
                    if not isinstance(fixture.get(key), str) or not fixture[key].strip():
                        errors.append(f"injected_fixture.{key} must be a non-empty string")
        elif item.get("injected_fixture") is not None:
            errors.append("injected_fixture must only be set for indirect_document_injection items")
    else:
        if item.get("attack_type") is not None:
            errors.append("attack_type must be null for non-adversarial items")
        if item.get("injected_fixture") is not None:
            errors.append("injected_fixture must only be set for indirect_document_injection items")

    # existence checks against the actual corpus, if provided
    if valid_document_ids is not None:
        for doc_id in item["gold_document_ids"]:
            if doc_id not in valid_document_ids:
                errors.append(f"gold_document_id not found in corpus: {doc_id}")
    if valid_chunk_ids is not None:
        for chunk_id in item["gold_chunk_ids"]:
            if chunk_id not in valid_chunk_ids:
                errors.append(f"gold_chunk_id not found in corpus: {chunk_id}")
    if chunk_id_to_document_id is not None:
        gold_docs = set(item["gold_document_ids"])
        for chunk_id in item["gold_chunk_ids"]:
            owner = chunk_id_to_document_id.get(chunk_id)
            if owner is not None and owner not in gold_docs:
                errors.append(
                    f"gold_chunk_id {chunk_id} belongs to document {owner}, "
                    f"which is not in gold_document_ids"
                )

    return errors


def validate_benchmark(
    items: Sequence[Dict],
    valid_chunk_ids: Optional[set] = None,
    valid_document_ids: Optional[set] = None,
    chunk_id_to_document_id: Optional[Dict[str, str]] = None,
) -> Dict[str, List]:
    """Validate a whole benchmark. Returns a dict:
        {"errors_by_id": {id: [errors]}, "duplicate_ids": [...]}
    Both empty means the benchmark is fully valid.
    """
    errors_by_id: Dict[str, List[str]] = {}
    seen_ids: Dict[str, int] = {}
    duplicate_ids: List[str] = []

    for item in items:
        item_id = item.get("id", "<missing-id>")
        seen_ids[item_id] = seen_ids.get(item_id, 0) + 1
        errs = validate_item(
            item, valid_chunk_ids, valid_document_ids, chunk_id_to_document_id
        )
        if errs:
            errors_by_id[item_id] = errs

    duplicate_ids = [i for i, count in seen_ids.items() if count > 1]

    return {"errors_by_id": errors_by_id, "duplicate_ids": duplicate_ids}
