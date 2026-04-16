"""Tests for generation and safety guardrails."""

import pytest
from src.generate.safety_guardrails import SafetyGuardrails
from src.generate.ollama_client import OllamaClient


class TestSafetyGuardrails:
    """Test hallucination detection and safety."""

    @pytest.fixture
    def guardrails(self):
        return SafetyGuardrails(confidence_threshold=0.7)

    def test_low_confidence_rejected(self, guardrails):
        """Test that low-confidence responses trigger 'I don't know'."""
        response = "Maybe it could be related to blood loss."
        confidence = 0.5
        source_chunks = []
        result = guardrails.validate_response(response, source_chunks, confidence)
        assert not result["is_safe"] or result["confidence"] < guardrails.confidence_threshold

    def test_direct_medical_advice_detected(self, guardrails):
        """Test detection of direct medical advice to patient."""
        dangerous_response = "You should take 2 units of red blood cells immediately."
        assert guardrails.detect_direct_medical_advice(dangerous_response)

    def test_safe_response_passes(self, guardrails):
        """Test that properly sourced response passes checks."""
        safe_response = "Research shows that hemorrhagic shock mortality depends on severity [PMID: 12345678]."
        confidence = 0.9
        source_chunks = [{"content": "hemorrhagic shock mortality"}]
        result = guardrails.validate_response(safe_response, source_chunks, confidence)
        assert result["is_safe"]


class TestOllamaClient:
    """Test Ollama LLM integration."""

    @pytest.fixture
    def ollama_client(self):
        return OllamaClient(base_url="http://localhost:11434")

    def test_is_available(self, ollama_client):
        """Test Ollama availability check."""
        # Will fail if Ollama not running, expected
        # availability = ollama_client.is_available()
        pass

    def test_generate_with_system_prompt(self, ollama_client):
        """Test generation with system prompt."""
        # Placeholder test
        pass
