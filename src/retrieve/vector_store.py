"""PostgreSQL + pgvector interface for document storage and retrieval."""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session


class VectorStore:
    """Vector store using PostgreSQL + pgvector.

    Performs cosine similarity search over the chunks table,
    with optional metadata filtering (source_type, year range).
    """

    def __init__(self, session: Session):
        self.session = session

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks by cosine distance.

        Args:
            query_embedding: Query vector (768-d for PubMedBERT)
            top_k: Number of results to return
            filters: Optional filters:
                - source_types: list[str] (e.g. ["pubmed", "fda"])
                - min_year: int (minimum document year)

        Returns:
            List of dicts with: chunk_id, document_id, text, score,
            source_type, source_id, title, year, url, metadata
        """
        vec_literal = "[" + ",".join(str(v) for v in query_embedding) + "]"

        where_clauses = ["c.embedding IS NOT NULL"]
        params: dict[str, Any] = {"top_k": top_k}

        if filters:
            if source_types := filters.get("source_types"):
                where_clauses.append("d.source_type = ANY(:source_types)")
                params["source_types"] = source_types
            if min_year := filters.get("min_year"):
                where_clauses.append("d.year >= :min_year")
                params["min_year"] = min_year

        where_sql = " AND ".join(where_clauses)

        sql = text(f"""
            SELECT
                c.id            AS chunk_id,
                c.document_id,
                c.text,
                c.token_count,
                c.metadata      AS chunk_metadata,
                d.source_type,
                d.source_id,
                d.title,
                d.year,
                d.url,
                1 - (c.embedding <=> CAST(:vec AS vector)) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {where_sql}
            ORDER BY c.embedding <=> CAST(:vec AS vector)
            LIMIT :top_k
        """)
        params["vec"] = vec_literal

        rows = self.session.execute(sql, params).mappings().all()

        return [
            {
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "text": r["text"],
                "token_count": r["token_count"],
                "chunk_metadata": r["chunk_metadata"],
                "source_type": r["source_type"],
                "source_id": r["source_id"],
                "title": r["title"],
                "year": r["year"],
                "url": r["url"],
                "score": float(r["score"]),
            }
            for r in rows
        ]

    def count_embedded(self) -> int:
        """Return number of chunks that have embeddings."""
        result = self.session.execute(
            text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        )
        return result.scalar()

    def count_total(self) -> int:
        """Return total number of chunks."""
        result = self.session.execute(text("SELECT COUNT(*) FROM chunks"))
        return result.scalar()
