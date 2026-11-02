"""Shared building blocks for the guarded query flow (Dimension 3).

api/rag_api.py (production) and scripts/evaluate_security.py (evaluation)
both build their LLM messages and apply the guardrails through THIS module, so
the thing that is evaluated is the thing that ships.

    [input guard] -> retrieval (AdaptiveRetriever, unchanged)
      -> [context sanitizer] -> build_context (unchanged)
      -> build_chat_messages [prompt hardening] -> LLM
      -> [output guard]

With guardrails=None (or SecurityConfig(enabled=False)) every function here
reproduces the pre-Dimension-3 behavior exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from generation.context_budget import build_context
from generation.prompt_templates import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from retrieval.adaptive import REFUSAL_MESSAGE as EVIDENCE_REFUSAL_MESSAGE
from security.guardrails import (
    SECURITY_CONTEXT_MESSAGE,
    SECURITY_INPUT_MESSAGE,
    Guardrails,
    with_canary,
)

# Stage names reported per query (stable strings, safe to aggregate on).
STAGE_INPUT_BLOCKED = 'input_blocked'
STAGE_EVIDENCE_REFUSED = 'evidence_refused'      # Dimension 2 refusal, not a security event
STAGE_NO_CHUNKS = 'no_chunks'
STAGE_CONTEXT_REFUSED = 'context_refused'        # every retrieved chunk was dropped
STAGE_GENERATED = 'generated'
STAGE_OUTPUT_BLOCKED = 'output_blocked'
STAGE_OUTPUT_REDACTED = 'output_redacted'


def _active(guardrails: Optional[Guardrails]) -> bool:
    return guardrails is not None and guardrails.enabled


def build_system_prompt(guardrails: Optional[Guardrails] = None,
                        base: Optional[str] = None) -> str:
    """The system prompt actually sent to the model."""
    prompt = RAG_SYSTEM_PROMPT if base is None else base
    if not _active(guardrails):
        return prompt
    prompt = with_canary(prompt, guardrails.config.canary)
    return guardrails.harden_system_prompt(prompt)


def build_chat_messages(question: str, context: str, history: Sequence[Dict],
                        guardrails: Optional[Guardrails] = None,
                        system_prompt: Optional[str] = None) -> List[Dict]:
    """Identical to the message list api/rag_api.py built before Dimension 3
    when guardrails are off; with guardrails on, the system prompt carries the
    security rules and the context is fenced as untrusted."""
    system = system_prompt if system_prompt is not None else build_system_prompt(guardrails)
    messages = [{'role': 'system', 'content': system}]
    for turn in list(history)[-3:]:
        messages.append({'role': 'user', 'content': turn['question']})
        messages.append({'role': 'assistant', 'content': turn['answer']})
    wrapped = guardrails.wrap_context(context) if _active(guardrails) else context
    messages.append({
        'role': 'user',
        'content': RAG_USER_TEMPLATE.format(context=wrapped, question=question),
    })
    return messages


@dataclass
class PipelineOutcome:
    answer: str
    stage: str
    chunks: List[Dict] = field(default_factory=list)
    messages: Optional[List[Dict]] = None
    events: List[Dict] = field(default_factory=list)   # one dict per guardrail that fired

    @property
    def generated(self) -> bool:
        return self.stage in (STAGE_GENERATED, STAGE_OUTPUT_BLOCKED, STAGE_OUTPUT_REDACTED)

    @property
    def security_blocked(self) -> bool:
        return self.stage in (STAGE_INPUT_BLOCKED, STAGE_CONTEXT_REFUSED, STAGE_OUTPUT_BLOCKED)


def run_guarded_query(
    question: str,
    *,
    retrieve: Callable[[str], object],
    generate: Callable[[List[Dict]], str],
    guardrails: Optional[Guardrails] = None,
    history: Sequence[Dict] = (),
    transform_chunks: Optional[Callable[[List[Dict]], List[Dict]]] = None,
    system_prompt_base: Optional[str] = None,
) -> PipelineOutcome:
    """The /query flow minus HTTP and session bookkeeping.

    `retrieve(question)` returns an AdaptiveResult-like object (`.refused`,
    `.chunks`). `generate(messages)` returns the model's answer text.
    `transform_chunks` exists only for evaluation: it is where a poisoned
    document is slipped into what retrieval returned.
    """
    active = _active(guardrails)
    events: List[Dict] = []

    if active:
        verdict = guardrails.check_input(question)
        if verdict.blocked:
            events.append(verdict.to_dict())
            return PipelineOutcome(SECURITY_INPUT_MESSAGE, STAGE_INPUT_BLOCKED, events=events)

    result = retrieve(question)
    if getattr(result, 'refused', False):
        return PipelineOutcome(EVIDENCE_REFUSAL_MESSAGE, STAGE_EVIDENCE_REFUSED, events=events)
    chunks = list(result.chunks)
    if transform_chunks is not None:
        chunks = transform_chunks(chunks)
    if not chunks:
        return PipelineOutcome('No relevant documents found for your question.',
                               STAGE_NO_CHUNKS, events=events)

    if active:
        chunks, report = guardrails.sanitize_chunks(chunks)
        if report.flagged:
            events.append(report.to_dict())
        if not chunks:
            return PipelineOutcome(SECURITY_CONTEXT_MESSAGE, STAGE_CONTEXT_REFUSED, events=events)

    max_chars = 3000 if history else 5000          # same budgets as api/rag_api.py
    context, _, _ = build_context(chunks, max_chars=max_chars)
    system = build_system_prompt(guardrails, base=system_prompt_base)
    messages = build_chat_messages(question, context, history, guardrails, system_prompt=system)
    answer = generate(messages)
    if not answer or len(answer.strip()) < 5:
        answer = 'I was unable to generate a response. Please try again.'

    stage = STAGE_GENERATED
    if active:
        out = guardrails.check_output(answer, system_prompt=system)
        if out.flagged:
            events.append(out.to_dict())
            answer = out.answer
            stage = STAGE_OUTPUT_BLOCKED if out.action == 'blocked' else STAGE_OUTPUT_REDACTED
    return PipelineOutcome(answer, stage, chunks=chunks, messages=messages, events=events)
