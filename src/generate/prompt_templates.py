"""Clinical-specific prompt templates for RAG generation.

Builds system and user prompts that ground LLM responses in retrieved evidence,
enforce citation discipline, and include few-shot examples for clinical Q&A.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are a clinical research assistant helping healthcare professionals \
find evidence-based information from medical literature.

RULES — follow every one:
1. Answer ONLY from the provided source documents. Do not use prior knowledge.
2. Cite every factual claim with the source tag shown in brackets, e.g. [1], [2].
3. If the sources do not contain enough information to answer, say \
"Insufficient evidence in the available sources to answer this question."
4. Never give direct patient-care instructions (e.g. "give the patient …"). \
Instead, summarise what the literature reports.
5. Note any conflicts or limitations across sources.
6. Be concise — clinicians need actionable summaries, not essays."""

FEW_SHOT_EXAMPLES = """\
--- Example 1 ---
Question: What triggers activation of a massive transfusion protocol?
Sources:
[1] (PMID 29451243 — MTP Guidelines, 2020) "Massive transfusion protocol \
is activated when a patient requires more than 10 units of packed red blood \
cells within 24 hours."
[2] (PMID 30120987 — Trauma Resuscitation, 2019) "Clinical triggers include \
systolic BP <90 mmHg, heart rate >120, and an Assessment of Blood Consumption \
(ABC) score ≥ 2."

Answer: Massive transfusion protocol (MTP) activation criteria include \
requirement of >10 units pRBCs in 24 hours [1] and clinical triggers such as \
SBP <90 mmHg, HR >120, or ABC score ≥ 2 [2]. Institutional protocols may vary.

--- Example 2 ---
Question: Should I give my patient TXA after a car accident?
Sources:
[1] (PMID 31200456 — TXA in Trauma, 2021) "Tranexamic acid administered \
within 3 hours of injury reduces mortality in hemorrhagic shock."

Answer: I cannot provide direct patient-care instructions. The literature \
reports that tranexamic acid (TXA) administered within 3 hours of injury \
is associated with reduced mortality in hemorrhagic shock [1]. Clinical \
decisions should follow your institution's MTP protocol and current \
ATLS/TCCC guidelines."""


def format_source_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered source block for the prompt.

    Each chunk dict should have at minimum: text.
    Optional keys used for richer citations: source_type, source_id, title, year.
    """
    lines: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        # Build a short header: "(source_type source_id — title, year)"
        parts: list[str] = []
        stype = chunk.get("source_type", "")
        sid = chunk.get("source_id", "")
        if stype and sid:
            label = f"PMID {sid}" if stype == "pubmed" else f"{stype.upper()} {sid}"
            parts.append(label)
        title = chunk.get("title", "")
        year = chunk.get("year")
        if title:
            parts.append(title + (f", {year}" if year else ""))
        header = f" ({' — '.join(parts)})" if parts else ""

        lines.append(f"[{idx}]{header} \"{chunk['text']}\"")

    return "\n".join(lines)


def build_user_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    """Assemble the full user-turn prompt with sources and question."""
    source_block = format_source_block(chunks)
    return (
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"--- Now answer the following ---\n"
        f"Sources:\n{source_block}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )


def build_messages(
    query: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return the full message list ready for an LLM chat-completion call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, chunks)},
    ]
