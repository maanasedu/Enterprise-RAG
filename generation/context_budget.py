"""Helpers for assembling LLM context strings within a character budget.

These replace ad-hoc raw character slicing (`context[:3000]`, `[:5000]`,
`[:6000]`), which could cut a retrieved chunk off mid-sentence, or even
mid-word, regardless of how highly that chunk was ranked. Instead, chunks
are appended whole, highest-ranked first (callers already hand this
function chunks in rank order), until adding the next one would exceed the
budget; any chunk that doesn't fit is dropped in full rather than sliced.
"""
from typing import Dict, List, Tuple

SOURCE_SEPARATOR = '\n\n---\n\n'


def build_context(chunks: List[Dict], max_chars: int) -> Tuple[str, int, bool]:
    """Assemble a context string from ranked chunks within `max_chars`.

    Args:
        chunks: retrieved chunks, best-ranked first (as returned by
            HybridRetriever.retrieve).
        max_chars: soft character budget for the assembled context.

    Returns:
        (context_string, chunks_included, was_truncated)

    Complete chunks are always preserved — no chunk is ever cut mid-way.
    The one exception is the highest-ranked chunk: if it alone is larger
    than `max_chars`, it is still included in full (an empty context is
    worse than a slightly over-budget one that at least contains the best
    match), and `was_truncated` reflects whether any lower-ranked chunks
    had to be dropped as a result.
    """
    if not chunks:
        return '', 0, False

    blocks = [
        f'[Source {i + 1}]\n{chunk.get("text", "")}'
        for i, chunk in enumerate(chunks)
    ]

    included: List[str] = []
    used_chars = 0

    for block in blocks:
        separator_len = len(SOURCE_SEPARATOR) if included else 0
        projected = used_chars + separator_len + len(block)

        if included and projected > max_chars:
            break

        included.append(block)
        used_chars = projected

        if len(included) == 1 and used_chars > max_chars:
            # The single best-ranked chunk alone exceeds the budget.
            # Keep it whole and stop rather than slicing it.
            break

    truncated = len(included) < len(blocks)
    return SOURCE_SEPARATOR.join(included), len(included), truncated
