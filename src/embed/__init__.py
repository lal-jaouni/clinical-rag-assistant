"""Embedding pipeline for clinical documents."""

from src.embed.models import EmbeddingModel
from src.embed.embeddings import batch_embed

__all__ = ["EmbeddingModel", "batch_embed"]
