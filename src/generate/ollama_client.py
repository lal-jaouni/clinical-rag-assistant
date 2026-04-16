"""Ollama LLM client for local inference."""

from typing import List, Dict, Any


class OllamaClient:
    """Ollama API client for local LLM inference.

    Supports Meditron-7B (primary) and BioMistral-7B (fallback) models.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        primary_model: str = "meditron:7b",
        fallback_model: str = "biomistral:7b",
        temperature: float = 0.2,
        max_tokens: int = 500,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            primary_model: Primary model name
            fallback_model: Fallback model if primary fails
            temperature: Generation temperature (low for clinical safety)
            max_tokens: Max tokens in response
        """
        self.base_url = base_url
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        use_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            use_fallback: Use fallback model if True

        Returns:
            Dict with keys: text, model, confidence, raw_response
        """
        # Implementation placeholder
        pass

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        # Implementation placeholder
        return True
