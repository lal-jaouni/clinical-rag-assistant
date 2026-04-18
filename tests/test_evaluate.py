"""Tests for evaluation metrics."""

import pytest
from src.evaluate.ragas_metrics import RAGASEvaluator
from src.evaluate.hallucination_detector import HallucinationDetector


class TestRAGASEvaluator:
    """Test RAGAS metrics."""

    @pytest.fixture
    def evaluator(self):
        return RAGASEvaluator()

    def test_faithfulness_scoring(self, evaluator):
        """Test faithfulness metric (answer grounded in context)."""
        answer = "Hemorrhagic shock mortality in Class III is 20-40%"
        context = ["Class III hemorrhage (30-40% blood loss) has mortality of 20-40%"]
        score = evaluator.faithfulness(answer, context)
        assert 0 <= score <= 1

    def test_answer_relevance(self, evaluator):
        """Test answer relevance to question."""
        question = "What is mortality in hemorrhagic shock?"
        answer = "Mortality depends on hemorrhage class and intervention timing."
        score = evaluator.answer_relevance(answer, question)
        assert 0 <= score <= 1

    def test_context_precision(self, evaluator):
        """Test context precision (relevant chunks)."""
        answer = "Hemorrhagic shock treatment"
        context = [
            "Hemorrhagic shock is caused by blood loss.",
            "Treatment involves fluid resuscitation.",
            "The color of hospitals varies by region.",  # Irrelevant
        ]
        score = evaluator.context_precision(answer, context)
        assert 0 <= score <= 1


class TestHallucinationDetector:
    """Test hallucination detection."""

    @pytest.fixture
    def detector(self):
        return HallucinationDetector(threshold=0.7)

    def test_detect_unsupported_fact(self, detector):
        """Test detection of fact not in sources."""
        answer = "The mortality rate is 99% and unicorns exist in hospitals."
        sources = ["Hemorrhagic shock mortality is 20-40% in Class III."]
        result = detector.detect_hallucinations(answer, sources)
        assert result["is_hallucinating"]

    def test_grounded_answer_passes(self, detector):
        """Test that grounded answer passes."""
        answer = "Class III hemorrhage has 20-40% mortality."
        sources = ["Class III hemorrhage (30-40% blood loss) has mortality of 20-40%."]
        result = detector.detect_hallucinations(answer, sources)
        assert not result["is_hallucinating"]

    def test_hallucination_rate_tracking(self, detector):
        """Test hallucination rate calculation."""
        detector.log_evaluation("answer 1", is_hallucinating=False)
        detector.log_evaluation("answer 2", is_hallucinating=True)
        detector.log_evaluation("answer 3", is_hallucinating=False)
        rate = detector.get_hallucination_rate()
        assert rate == pytest.approx(0.33, abs=0.01)
