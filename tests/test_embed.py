"""Tests for embedding pipeline."""

import pytest
from src.embed.models import EmbeddingModel


class TestEmbeddingModel:
    """Test embedding model."""

    @pytest.fixture
    def embedding_model(self):
        return EmbeddingModel(model_name="pubmedbert-base-uncased-abstract", device="cpu")

    def test_embed_single_text(self, embedding_model):
        """Test embedding a single text."""
        # Placeholder test
        pass

    def test_embed_batch(self, embedding_model):
        """Test batch embedding."""
        texts = [
            "Hemorrhagic shock is a critical condition.",
            "Transfusion protocols vary by institution.",
        ]
        # Placeholder test
        pass

    def test_embedding_dimension(self, embedding_model):
        """Test that embeddings have correct dimensionality."""
        # PubMedBERT should produce 768-dimensional vectors
        pass
