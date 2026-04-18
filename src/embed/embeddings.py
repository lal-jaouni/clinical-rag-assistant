"""Batch embedding pipeline: reads chunks from DB, embeds, writes vectors back."""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy.orm import Session

from embed.models import EmbeddingModel

logger = logging.getLogger(__name__)


def batch_embed_db(
    session: Session,
    embedding_model: EmbeddingModel,
    batch_size: int = 32,
    only_missing: bool = True,
) -> dict[str, Any]:
    """Embed all chunks in the database, writing vectors directly to chunks.embedding.

    Args:
        session: SQLAlchemy session (caller must close)
        embedding_model: Initialized EmbeddingModel
        batch_size: Number of chunks to encode at once
        only_missing: If True, skip chunks that already have an embedding

    Returns:
        Summary dict with counts and timing
    """
    from db.schema import Chunk

    start = time.time()
    summary = {"total_chunks": 0, "embedded": 0, "skipped": 0, "elapsed_seconds": 0.0}

    # Query chunks to embed
    query = session.query(Chunk)
    if only_missing:
        query = query.filter(Chunk.embedding.is_(None))
    query = query.order_by(Chunk.id)

    chunks = query.all()
    summary["total_chunks"] = len(chunks)

    if not chunks:
        summary["elapsed_seconds"] = time.time() - start
        return summary

    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]

        vectors = embedding_model.embed(texts, batch_size=batch_size)

        for chunk, vec in zip(batch, vectors):
            chunk.embedding = vec
            summary["embedded"] += 1

        session.flush()

        done = min(i + batch_size, len(chunks))
        logger.info(f"[{done}/{len(chunks)}] Embedded {done} chunks")

    session.commit()
    summary["elapsed_seconds"] = time.time() - start
    return summary


def batch_embed_texts(
    texts: list[str],
    embedding_model: EmbeddingModel,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed a list of texts without DB interaction.

    Useful for embedding query-time texts or test data.
    """
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_vectors.extend(embedding_model.embed(batch, batch_size=batch_size))
    return all_vectors
