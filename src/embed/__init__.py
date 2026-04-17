"""Embedding model wrappers (PubMedBERT, BioBERT, MedCPT) and batch pipeline."""

from src.embed.models import EmbeddingModel
from src.embed.embeddings import batch_embed_db, batch_embed_texts

__all__ = ["EmbeddingModel", "batch_embed_db", "batch_embed_texts"]
