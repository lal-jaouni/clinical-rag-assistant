"""Hallucination detection, confidence thresholding, and clinical safety guardrails.

Three layers of defence:
1. **Confidence gate** — if the LLM's self-reported or heuristic confidence is
   below threshold, the answer is replaced with a safe refusal.
2. **Source-grounding check** — verifies that key n-grams in the answer appear
   in the retrieved source chunks (lexical overlap proxy for faithfulness).
3. **Scope guard** — detects direct patient-care instructions and flags them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface for an embedding model (e.g. EmbeddingModel)."""

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]: ...


# Phrases that indicate the model is giving direct patient instructions
_DIRECT_ADVICE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\byou\s+should\s+(take|give|administer|inject|prescribe)\b",
        r"\bgive\s+(the|your)\s+patient\b",
        r"\badminister\s+\d+\s*(mg|ml|units|mcg)\b",
        r"\bstart\s+(the|your)\s+patient\s+on\b",
        r"\bi\s+recommend\s+(you|the patient)\s+(take|start|stop)\b",
        r"\btake\s+\d+\s*(mg|ml|tablets?|capsules?)\b",
    ]
]

# Phrases that signal the model is hedging / uncertain
_UNCERTAINTY_PHRASES = [
    "i don't know",
    "i do not know",
    "insufficient evidence",
    "cannot determine",
    "not enough information",
    "unable to answer",
    "no relevant sources",
    "the sources do not",
    "the provided sources do not",
]

SAFE_REFUSAL = (
    "Insufficient evidence in the available sources to answer this question. "
    "Please consult primary literature or a subject-matter expert."
)


@dataclass
class SafetyResult:
    """Outcome of running a response through all safety checks."""

    is_safe: bool
    final_response: str
    confidence: float
    grounding_score: float
    has_citations: bool
    has_direct_advice: bool
    has_uncertainty: bool
    override_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class SafetyGuardrails:
    """Multi-layer safety validation for generated clinical answers."""

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        grounding_threshold: float = 0.50,
        min_citation_ratio: float = 0.3,
        embedding_model: Embedder | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.grounding_threshold = grounding_threshold
        self.min_citation_ratio = min_citation_ratio
        self.embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        response: str,
        source_chunks: list[dict[str, Any]],
        confidence: float | None = None,
    ) -> SafetyResult:
        """Run all safety checks and return a SafetyResult.

        If ``confidence`` is None it is estimated heuristically from the response
        text and grounding score.
        """
        grounding = self.check_source_grounding(response, source_chunks)
        has_citations = self._has_citations(response)
        has_advice = self.detect_direct_medical_advice(response)
        has_uncertainty = self._detect_uncertainty(response)

        # Estimate confidence if not provided by the LLM
        if confidence is None:
            confidence = self._estimate_confidence(
                response, grounding, has_citations, has_uncertainty
            )

        # Decision logic
        override_reason: str | None = None
        final_response = response

        if has_uncertainty:
            # Model already refused — that's safe, keep it
            override_reason = None
        elif confidence < self.confidence_threshold:
            override_reason = f"confidence {confidence:.2f} < threshold {self.confidence_threshold}"
            final_response = SAFE_REFUSAL
        elif grounding < self.grounding_threshold and not has_uncertainty:
            override_reason = (
                f"grounding {grounding:.2f} < threshold {self.grounding_threshold}"
            )
            final_response = SAFE_REFUSAL
        elif has_advice:
            override_reason = "direct medical advice detected"
            final_response = (
                "Note: This response was flagged for containing direct patient-care "
                "instructions. The literature reports the following:\n\n" + response
            )

        is_safe = override_reason is None

        return SafetyResult(
            is_safe=is_safe,
            final_response=final_response,
            confidence=confidence,
            grounding_score=grounding,
            has_citations=has_citations,
            has_direct_advice=has_advice,
            has_uncertainty=has_uncertainty,
            override_reason=override_reason,
            details={
                "confidence_threshold": self.confidence_threshold,
                "grounding_threshold": self.grounding_threshold,
            },
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_source_grounding(
        self, response: str, source_chunks: list[dict[str, Any]]
    ) -> float:
        """Score how well the response is grounded in the sources (0-1).

        When an embedding model is available, uses cosine similarity between
        the response embedding and each source chunk embedding (max score).
        This handles paraphrasing far better than lexical overlap.

        Falls back to n-gram overlap when no embedding model is set.
        """
        if not source_chunks or not response.strip():
            return 0.0

        if self.embedding_model is not None:
            return self._semantic_grounding(response, source_chunks)
        return self._ngram_grounding(response, source_chunks)

    def _semantic_grounding(
        self, response: str, source_chunks: list[dict[str, Any]]
    ) -> float:
        """Cosine similarity between response and source chunk embeddings.

        Embeds the response and each source chunk, returns the max pairwise
        cosine similarity. PubMedBERT embeddings are L2-normalized, so
        cosine similarity = dot product.
        """
        source_texts = [c.get("text", "") for c in source_chunks if c.get("text")]
        if not source_texts:
            return 0.0

        # Strip bracket citations before embedding the response
        clean_response = re.sub(r"\[\d+\]", "", response).strip()

        all_texts = [clean_response] + source_texts
        embeddings = self.embedding_model.embed(all_texts)

        response_emb = embeddings[0]
        source_embs = embeddings[1:]

        # Max cosine similarity across source chunks
        max_sim = max(
            sum(a * b for a, b in zip(response_emb, source_emb))
            for source_emb in source_embs
        )
        # Clamp to [0, 1]
        return max(0.0, min(1.0, max_sim))

    def _ngram_grounding(
        self, response: str, source_chunks: list[dict[str, Any]]
    ) -> float:
        """Fallback: n-gram overlap grounding score (0-1).

        Uses overlapping content-word n-grams (n=2,3) between the answer and
        the concatenated source texts as a lightweight faithfulness proxy.
        """
        source_text = " ".join(c.get("text", "") for c in source_chunks).lower()

        # Extract content-word n-grams from the response (skip stopwords, citations)
        response_clean = re.sub(r"\[\d+\]", "", response.lower())
        response_words = _content_words(response_clean)

        if len(response_words) < 3:
            return 1.0  # very short answer — can't meaningfully score

        # Build 2-gram and 3-gram sets from source
        source_words = _content_words(source_text)
        source_bigrams = _ngrams(source_words, 2)
        source_trigrams = _ngrams(source_words, 3)
        source_ng = source_bigrams | source_trigrams

        resp_bigrams = _ngrams(response_words, 2)
        resp_trigrams = _ngrams(response_words, 3)
        resp_ng = resp_bigrams | resp_trigrams

        if not resp_ng:
            return 1.0

        overlap = len(resp_ng & source_ng)
        return overlap / len(resp_ng)

    @staticmethod
    def detect_direct_medical_advice(text: str) -> bool:
        """Return True if the text contains direct patient-care instructions."""
        return any(pat.search(text) for pat in _DIRECT_ADVICE_PATTERNS)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_citations(text: str) -> bool:
        """Check if the response contains bracket citations like [1], [2]."""
        return bool(re.search(r"\[\d+\]", text))

    @staticmethod
    def _detect_uncertainty(text: str) -> bool:
        lower = text.lower()
        return any(phrase in lower for phrase in _UNCERTAINTY_PHRASES)

    def _estimate_confidence(
        self,
        response: str,
        grounding: float,
        has_citations: bool,
        has_uncertainty: bool,
    ) -> float:
        """Heuristic confidence when the LLM doesn't provide one."""
        if has_uncertainty:
            return 0.3
        score = 0.5  # baseline
        score += 0.25 * grounding  # up to +0.25 for perfect grounding
        if has_citations:
            score += 0.15
        if len(response.split()) > 20:
            score += 0.05
        return min(score, 1.0)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very and "
    "but if or because until while about also that this these those it its".split()
)


def _content_words(text: str) -> list[str]:
    """Extract non-stopword alphabetic tokens."""
    return [w for w in re.findall(r"[a-z]{3,}", text) if w not in _STOPWORDS]


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}
