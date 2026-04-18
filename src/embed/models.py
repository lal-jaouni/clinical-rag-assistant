"""Embedding model selection and initialization.

Wraps sentence-transformers for clinical embedding models (PubMedBERT, BioBERT, MedCPT).
Lazy-loads the model on first use to avoid slow imports at startup.
"""

from __future__ import annotations

from typing import List

# HuggingFace model IDs for the short names used in configs/model.yaml
MODEL_ALIASES: dict[str, str] = {
    "pubmedbert-base-uncased-abstract": "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "biobert-v1.1": "dmis-lab/biobert-v1.1",
    "medcpt-article": "ncbi/MedCPT-Article-Encoder",
    "medcpt-query": "ncbi/MedCPT-Query-Encoder",
}


class EmbeddingModel:
    """Clinical embedding model wrapper.

    Supports PubMedBERT and BioBERT for domain-specific semantic understanding.
    Uses sentence-transformers for encoding; lazy-loads the model on first call.
    """

    def __init__(self, model_name: str = "pubmedbert-base-uncased-abstract", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        hf_id = MODEL_ALIASES.get(self.model_name, self.model_name)
        self._model = SentenceTransformer(hf_id, device=self.device)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        self._load_model()
        return self._model.get_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> List[List[float]]:
        """Embed a list of texts to vectors.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show tqdm progress bar

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.

        For models with asymmetric query/document encoding (e.g. MedCPT),
        this would use the query encoder. For symmetric models like PubMedBERT,
        it's equivalent to embed([query])[0].
        """
        return self.embed([query])[0]
