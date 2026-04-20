"""Cross-encoder reranking for retrieval precision.

Disabled by default (configs/retrieval.yaml: reranker.enabled = false).
When enabled, takes the top-N results from HybridRetriever and re-scores
each (query, chunk) pair through a cross-encoder for more accurate ranking.
"""

from __future__ import annotations

from typing import Any


class Reranker:
    """Rerank retrieved documents using a cross-encoder model.

    Lazy-loads the model on first call to avoid import cost when disabled.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Rerank documents by relevance to query.

        Args:
            query: Natural language query
            documents: List of result dicts (must have 'text' key)
            top_k: Return top-k after reranking
            threshold: Minimum score to include

        Returns:
            Reranked documents above threshold, with 'rerank_score' added
        """
        if not documents:
            return []

        self._load_model()

        pairs = [(query, doc["text"]) for doc in documents]
        scores = self._model.predict(pairs)

        scored = []
        for doc, score in zip(documents, scores):
            s = float(score)
            if s >= threshold:
                scored.append({**doc, "rerank_score": s})

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]
