"""Response formatting with source attribution, citation links, and metadata.

Transforms raw LLM output + retrieved chunks into a structured response dict
suitable for API responses or UI rendering.
"""

from __future__ import annotations

import re
from typing import Any


def build_citation_link(source: dict[str, Any]) -> str:
    """Build a human-readable citation string with URL when possible.

    Examples:
        "[PMID: 29451243]" with url "https://pubmed.ncbi.nlm.nih.gov/29451243"
        "[FDA fda-2021-D-1234]"
        "[CT NCT04123456]"
    """
    stype = source.get("source_type", "unknown")
    sid = source.get("source_id", "")

    if stype == "pubmed":
        label = f"PMID: {sid}"
        url = f"https://pubmed.ncbi.nlm.nih.gov/{sid}"
    elif stype == "fda":
        label = f"FDA {sid}"
        url = source.get("url", "")
    elif stype == "clinical_trials":
        label = f"CT {sid}"
        url = f"https://clinicaltrials.gov/study/{sid}" if sid else ""
    else:
        label = sid or stype
        url = source.get("url", "")

    return {"label": f"[{label}]", "url": url}


def build_source_card(idx: int, chunk: dict[str, Any]) -> dict[str, Any]:
    """Build a structured source card for one retrieved chunk."""
    citation = build_citation_link(chunk)
    return {
        "index": idx,
        "citation": citation["label"],
        "url": citation["url"],
        "title": chunk.get("title", ""),
        "year": chunk.get("year"),
        "source_type": chunk.get("source_type", "unknown"),
        "source_id": chunk.get("source_id", ""),
        "text_preview": _truncate(chunk.get("text", ""), 200),
        "relevance_score": chunk.get("rrf_score") or chunk.get("similarity") or None,
    }


def format_response(
    answer: str,
    source_chunks: list[dict[str, Any]],
    confidence: float,
    grounding_score: float,
    latency_ms: int | None = None,
    model: str | None = None,
    is_safe: bool = True,
    override_reason: str | None = None,
) -> dict[str, Any]:
    """Assemble the final structured response.

    The returned dict is JSON-serialisable and used by the API layer.

    Keys:
        answer          — final answer text (may be a safe refusal)
        confidence      — 0-1 confidence score
        grounding_score — 0-1 source overlap score
        is_safe         — whether the response passed all safety checks
        override_reason — why the response was overridden (None if safe)
        sources         — list of source cards with citations and URLs
        cited_indices   — which source indices [1], [2] … appear in the answer
        latency_ms      — end-to-end generation latency
        model           — LLM model used
    """
    sources = [build_source_card(i + 1, c) for i, c in enumerate(source_chunks)]
    cited = _extract_cited_indices(answer)

    return {
        "answer": answer,
        "confidence": round(confidence, 3),
        "grounding_score": round(grounding_score, 3),
        "is_safe": is_safe,
        "override_reason": override_reason,
        "sources": sources,
        "cited_indices": cited,
        "latency_ms": latency_ms,
        "model": model,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_cited_indices(text: str) -> list[int]:
    """Pull bracket citation numbers from the answer, deduplicated and sorted."""
    return sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", text)))


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rsplit(" ", 1)[0] + "..."
