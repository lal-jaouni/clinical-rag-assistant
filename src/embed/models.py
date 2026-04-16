"""Embedding model selection and initialization."""

from typing import List


class EmbeddingModel:
    """Clinical embedding model wrapper.

    Supports PubMedBERT and BioBERT for domain-specific semantic understanding.
    """

    def __init__(self, model_name: str = "pubmedbert-base-uncased-abstract", device: str = "cpu"):
        """Initialize embedding model.

        Args:
            model_name: HuggingFace model ID
            device: torch device (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.device = device
        # Model loading deferred to lazy initialization

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts to vectors.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        # Implementation placeholder
        pass

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query (may use different strategy than documents)."""
        # Implementation placeholder
        pass
