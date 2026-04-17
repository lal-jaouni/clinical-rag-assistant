"""Hybrid retrieval combining vector search and BM25 keyword matching.

Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers
into a single ranked list.
"""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieve.vector_store import VectorStore


class BM25Index:
    """In-memory BM25 index built from chunk texts in the database."""

    def __init__(self, corpus: list[dict[str, Any]]):
        """Build BM25 index from a list of chunk dicts.

        Args:
            corpus: List of dicts with at least 'chunk_id' and 'text' keys.
        """
        self._corpus = corpus
        self._id_to_idx = {c["chunk_id"]: i for i, c in enumerate(corpus)}
        tokenized = [c["text"].lower().split() for c in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k chunks by BM25 score.

        Returns list of dicts with chunk_id, text, and bm25_score.
        """
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        scored = [(self._corpus[i], float(scores[i])) for i in range(len(self._corpus))]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored[:top_k]:
            results.append({**chunk, "bm25_score": score})
        return results

    @classmethod
    def from_db(cls, session) -> "BM25Index":
        """Build a BM25 index from all chunks in the database."""
        from sqlalchemy import text as sql_text

        rows = session.execute(
            sql_text("""
                SELECT c.id AS chunk_id, c.document_id, c.text, c.token_count,
                       c.metadata AS chunk_metadata,
                       d.source_type, d.source_id, d.title, d.year, d.url
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.id
            """)
        ).mappings().all()

        corpus = [dict(r) for r in rows]
        return cls(corpus)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum over lists of 1 / (k + rank_in_list).
    Default k=60 per the original Cormack et al. paper.

    Args:
        ranked_lists: Each list is an ordered list of result dicts (must have 'chunk_id').
        k: RRF constant (higher = more weight to lower-ranked results).

    Returns:
        Merged list sorted by RRF score descending, with 'rrf_score' added.
    """
    scores: dict[int, float] = {}
    docs: dict[int, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in docs:
                docs[cid] = doc

    fused = []
    for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        fused.append({**docs[cid], "rrf_score": score})
    return fused


class HybridRetriever:
    """Hybrid retriever combining semantic (pgvector) and keyword (BM25) retrieval.

    Fuses results via Reciprocal Rank Fusion for robust clinical retrieval.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index | None = None,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        fusion_method: str = "rrf",
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.fusion_method = fusion_method

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        vector_top_k: int | None = None,
        bm25_top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents using hybrid approach.

        Args:
            query: Natural language query (for BM25)
            query_embedding: Query embedding vector (for vector search)
            top_k: Number of final results after fusion
            vector_top_k: Override vector search top-k (default: top_k * 2)
            bm25_top_k: Override BM25 top-k (default: top_k * 2)
            filters: Metadata filters for vector search

        Returns:
            Fused list of top-k documents with combined scores
        """
        v_k = vector_top_k or top_k * 2
        b_k = bm25_top_k or top_k * 2

        # Vector search
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=v_k,
            filters=filters,
        )

        # BM25 search (if index available)
        if self.bm25_index is not None:
            bm25_results = self.bm25_index.search(query, top_k=b_k)
            ranked_lists = [vector_results, bm25_results]
        else:
            ranked_lists = [vector_results]

        # Fuse with RRF
        fused = reciprocal_rank_fusion(ranked_lists)
        return fused[:top_k]
