"""Tests for embedding pipeline.

These tests use a lightweight model (all-MiniLM-L6-v2) to avoid downloading
the full PubMedBERT model in CI. The interface is the same.
"""

import pytest
from src.embed.models import EmbeddingModel, MODEL_ALIASES
from src.embed.embeddings import batch_embed_texts


# Use a small, fast model for tests (downloads ~80MB vs ~400MB for PubMedBERT)
TEST_MODEL = "all-MiniLM-L6-v2"


class TestEmbeddingModel:
    @pytest.fixture
    def model(self):
        return EmbeddingModel(model_name=TEST_MODEL, device="cpu")

    def test_embed_single_text(self, model):
        texts = ["Hemorrhagic shock is a critical condition."]
        result = model.embed(texts)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) > 0
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_batch(self, model):
        texts = [
            "Hemorrhagic shock is a critical condition.",
            "Transfusion protocols vary by institution.",
            "Damage control resuscitation focuses on permissive hypotension.",
        ]
        result = model.embed(texts)
        assert len(result) == 3
        # All embeddings should have the same dimension
        dims = {len(v) for v in result}
        assert len(dims) == 1

    def test_embed_query(self, model):
        vec = model.embed_query("What is massive transfusion protocol?")
        assert isinstance(vec, list)
        assert len(vec) == model.dimension

    def test_embedding_dimension(self, model):
        dim = model.dimension
        assert isinstance(dim, int)
        assert dim > 0

    def test_normalized_embeddings(self, model):
        """Embeddings should be L2-normalized (unit vectors)."""
        vec = model.embed_query("test query")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_model_aliases(self):
        assert "pubmedbert-base-uncased-abstract" in MODEL_ALIASES
        assert "biobert-v1.1" in MODEL_ALIASES


class TestBatchEmbedTexts:
    @pytest.fixture
    def model(self):
        return EmbeddingModel(model_name=TEST_MODEL, device="cpu")

    def test_batch_embed_texts(self, model):
        texts = ["text one", "text two", "text three", "text four", "text five"]
        result = batch_embed_texts(texts, model, batch_size=2)
        assert len(result) == 5
        assert all(len(v) == model.dimension for v in result)

    def test_empty_input(self, model):
        result = batch_embed_texts([], model)
        assert result == []
