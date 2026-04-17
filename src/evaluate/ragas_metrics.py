"""RAGAS-style evaluation metrics for RAG systems.

Embedding-free implementation using content-word overlap as a lightweight
proxy. Suitable for offline evaluation without LLM calls.

Metrics:
- faithfulness: How well the answer is grounded in retrieved context.
- answer_relevance: How relevant the answer is to the question.
- context_precision: Fraction of retrieved chunks that contribute to the answer.
"""

from __future__ import annotations

import re
from typing import Any


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


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class RAGASEvaluator:
    """RAGAS (Retrieval-Augmented Generation Assessment) metrics.

    Implements faithfulness, answer_relevance, context_precision metrics
    to evaluate clinical RAG quality.
    """

    def __init__(self, llm_client=None):
        """Initialize RAGAS evaluator.

        Args:
            llm_client: Reserved for future LLM-based evaluation. Currently unused.
        """
        self.llm_client = llm_client

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate RAG outputs using RAGAS metrics.

        Args:
            questions: List of clinical questions.
            answers: List of generated answers.
            contexts: List of context lists (retrieved chunks per question).
            ground_truths: Optional gold-standard answers (unused currently).

        Returns:
            Dict with aggregate and per-question metric scores.
        """
        n = len(questions)
        if not (n == len(answers) == len(contexts)):
            raise ValueError("questions, answers, and contexts must have equal length")

        faith_scores = []
        relevance_scores = []
        precision_scores = []

        per_question: list[dict[str, float]] = []

        for i in range(n):
            f = self.faithfulness(answers[i], contexts[i])
            r = self.answer_relevance(answers[i], questions[i])
            p = self.context_precision(answers[i], contexts[i])
            faith_scores.append(f)
            relevance_scores.append(r)
            precision_scores.append(p)
            per_question.append({
                "faithfulness": round(f, 4),
                "answer_relevance": round(r, 4),
                "context_precision": round(p, 4),
            })

        def _mean(vals: list[float]) -> float:
            return sum(vals) / max(len(vals), 1)

        return {
            "faithfulness": round(_mean(faith_scores), 4),
            "answer_relevance": round(_mean(relevance_scores), 4),
            "context_precision": round(_mean(precision_scores), 4),
            "num_questions": n,
            "per_question": per_question,
        }

    @staticmethod
    def faithfulness(answer: str, context: list[str]) -> float:
        """Score answer faithfulness to context (0-1).

        Measures what fraction of the answer's content-word bigrams and trigrams
        appear in the combined context. High overlap = well-grounded answer.
        """
        combined = " ".join(context)
        answer_clean = re.sub(r"\[\d+\]", "", answer)

        answer_words = _content_words(answer_clean)
        context_words = _content_words(combined)

        if len(answer_words) < 3:
            return 1.0  # too short to meaningfully score

        # Build n-gram sets
        ans_ng = _ngrams(answer_words, 2) | _ngrams(answer_words, 3)
        ctx_ng = _ngrams(context_words, 2) | _ngrams(context_words, 3)

        if not ans_ng:
            return 1.0

        overlap = len(ans_ng & ctx_ng)
        return overlap / len(ans_ng)

    @staticmethod
    def answer_relevance(answer: str, question: str) -> float:
        """Score answer relevance to question (0-1).

        Measures content-word overlap between the answer and question.
        A relevant answer should share key clinical terms with the question.
        """
        answer_words = set(_content_words(re.sub(r"\[\d+\]", "", answer)))
        question_words = set(_content_words(question))

        if not question_words:
            return 1.0

        # What fraction of question terms appear in the answer
        overlap = len(answer_words & question_words)
        return overlap / len(question_words)

    @staticmethod
    def context_precision(answer: str, context: list[str]) -> float:
        """Score context precision (0-1).

        Measures what fraction of retrieved context chunks are relevant to the
        answer. A high score means the retriever returned focused, useful chunks
        rather than noise.
        """
        if not context:
            return 0.0

        answer_words = set(_content_words(re.sub(r"\[\d+\]", "", answer)))
        if not answer_words:
            return 0.0

        relevant_count = 0
        relevance_threshold = 0.15  # at least 15% word overlap

        for chunk in context:
            chunk_words = set(_content_words(chunk))
            if not chunk_words:
                continue
            overlap = len(answer_words & chunk_words) / len(chunk_words)
            if overlap >= relevance_threshold:
                relevant_count += 1

        return relevant_count / len(context)

    def __repr__(self) -> str:
        return f"RAGASEvaluator(llm_client={'set' if self.llm_client else 'none'})"
