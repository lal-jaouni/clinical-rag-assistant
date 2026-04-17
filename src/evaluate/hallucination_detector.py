"""Hallucination detection and rate tracking.

Splits generated answers into sentences and checks each against source chunks
using content-word overlap. Sentences with low grounding are flagged as
potential hallucinations. Target: <2% hallucination rate across evaluations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# Reuse the same stopword set as safety_guardrails for consistency
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very and "
    "but if or because until while about also that this these those it its".split()
)


def _content_words(text: str) -> list[str]:
    """Extract non-stopword alphabetic tokens (3+ chars)."""
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOPWORDS]


def _word_overlap(words_a: list[str], words_b_set: set[str]) -> float:
    """Fraction of words_a that appear in words_b_set."""
    if not words_a:
        return 1.0
    return sum(1 for w in words_a if w in words_b_set) / len(words_a)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on period, semicolon, or newline boundaries."""
    parts = re.split(r"[.;!\n]+", text)
    return [s.strip() for s in parts if s.strip() and len(s.strip()) > 5]


class HallucinationDetector:
    """Detect and track hallucinations in generated answers.

    Measures the rate of facts in generated answers that cannot be grounded
    in retrieved documents. Target: <2% hallucination rate.
    """

    def __init__(self, threshold: float = 0.7):
        """Initialize detector.

        Args:
            threshold: Content-word overlap ratio below which a sentence
                       is considered ungrounded (0-1).
        """
        self.threshold = threshold
        self.hallucination_log: list[dict[str, Any]] = []

    def detect_hallucinations(
        self,
        answer: str,
        source_chunks: list[str | dict[str, Any]],
    ) -> dict[str, Any]:
        """Detect hallucinations in answer by checking sentence-level grounding.

        Args:
            answer: Generated answer text.
            source_chunks: Retrieved source texts (strings or dicts with "text" key).

        Returns:
            Dict with keys: is_hallucinating, hallucination_rate, flagged_facts
        """
        # Normalize source chunks to plain strings
        source_texts: list[str] = []
        for chunk in source_chunks:
            if isinstance(chunk, dict):
                source_texts.append(chunk.get("text", ""))
            else:
                source_texts.append(str(chunk))

        combined_source = " ".join(source_texts)
        source_word_set = set(_content_words(combined_source))

        # Strip bracket citations before analysis
        clean_answer = re.sub(r"\[\d+\]", "", answer).strip()

        # Whole-answer overlap as primary signal
        answer_words = _content_words(clean_answer)
        overall_overlap = _word_overlap(answer_words, source_word_set)

        # Sentence-level analysis for flagged_facts
        sentences = _split_sentences(clean_answer)
        flagged: list[str] = []
        sentence_scores: list[float] = []

        for sentence in sentences:
            words = _content_words(sentence)
            score = _word_overlap(words, source_word_set)
            sentence_scores.append(score)
            if score < self.threshold:
                flagged.append(sentence)

        # Hallucination rate = fraction of sentences that are ungrounded
        if sentences:
            hallucination_rate = len(flagged) / len(sentences)
        else:
            hallucination_rate = 0.0

        is_hallucinating = overall_overlap < self.threshold

        return {
            "is_hallucinating": is_hallucinating,
            "hallucination_rate": hallucination_rate,
            "flagged_facts": flagged,
            "overall_grounding": round(overall_overlap, 4),
            "sentence_scores": [round(s, 4) for s in sentence_scores],
        }

    def log_evaluation(
        self,
        answer: str,
        is_hallucinating: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an evaluation result for aggregate tracking."""
        self.hallucination_log.append({
            "answer_preview": answer[:200],
            "is_hallucinating": is_hallucinating,
            "details": details or {},
        })

    def get_hallucination_rate(self) -> float:
        """Get overall hallucination rate across all logged evaluations."""
        if not self.hallucination_log:
            return 0.0
        flagged = sum(1 for e in self.hallucination_log if e["is_hallucinating"])
        return flagged / len(self.hallucination_log)

    def save_report(self, path: str | Path = "metrics/hallucination_report.json") -> None:
        """Save hallucination tracking report to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "total_evaluations": len(self.hallucination_log),
            "hallucination_rate": round(self.get_hallucination_rate(), 4),
            "hallucinating_count": sum(
                1 for e in self.hallucination_log if e["is_hallucinating"]
            ),
            "target_rate": 0.02,
            "meets_target": self.get_hallucination_rate() <= 0.02,
            "evaluations": self.hallucination_log,
        }

        with open(out, "w") as f:
            json.dump(report, f, indent=2)

    def __repr__(self) -> str:
        n = len(self.hallucination_log)
        rate = self.get_hallucination_rate()
        return f"HallucinationDetector(threshold={self.threshold}, logged={n}, rate={rate:.2%})"
