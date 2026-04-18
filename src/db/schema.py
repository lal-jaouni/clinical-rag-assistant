"""SQLAlchemy models for clinical RAG.

Kept in sync with scripts/init_db.sql. The SQL file bootstraps a fresh DB;
these models are for application-side ORM reads and writes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Default dim = PubMedBERT (768). Change if swapping embedding models.
EMBEDDING_DIM = 768


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One source document before chunking (PubMed article, FDA PDF, etc.)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_id: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer)
    doi: Mapped[str | None] = mapped_column(String)
    url: Mapped[str | None] = mapped_column(String)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list[Chunk]] = relationship(back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("source_type", "source_id", name="documents_source_unique"),)


class Chunk(Base):
    """A retrieval-sized piece of a document, with an embedding vector."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="chunks_document_position_unique"),
    )


class EvalLog(Base):
    """One row per query during evaluation runs (tracks hallucination, latency, etc.)."""

    __tablename__ = "eval_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    cited_sources: Mapped[list[str] | None] = mapped_column(ARRAY(String))
    confidence: Mapped[float | None] = mapped_column(Float)
    hallucination: Mapped[bool | None] = mapped_column(Boolean)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    llm_provider: Mapped[str | None] = mapped_column(String)
    llm_model: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
