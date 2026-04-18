"""End-to-end RAG pipeline: query -> retrieve -> generate -> validate -> format.

Usage:
    from generate.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline.from_config(config)
    result = pipeline.answer("What triggers MTP activation?")
"""

from __future__ import annotations

import time
from typing import Any

from generate.litellm_client import LLMClient, LLMResponse
from generate.output_formatter import format_response
from generate.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from generate.safety_guardrails import Embedder, SafetyGuardrails, SafetyResult


class RAGPipeline:
    """Chains retrieval, generation, safety validation, and formatting."""

    def __init__(
        self,
        llm: LLMClient,
        retriever,
        guardrails: SafetyGuardrails,
        embedding_model=None,
        query_processor=None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.guardrails = guardrails
        self.embedding_model = embedding_model
        self.query_processor = query_processor

        # Share embedding model with guardrails for semantic grounding.
        # Only auto-share real Embedder instances (not mocks).
        if (
            embedding_model is not None
            and guardrails.embedding_model is None
            and isinstance(embedding_model, Embedder)
        ):
            guardrails.embedding_model = embedding_model

    def answer(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Full pipeline: query -> retrieve -> generate -> safety -> format.

        Args:
            query: Clinical question from the user.
            top_k: Number of chunks to retrieve.
            filters: Optional metadata filters (source_types, min_year).

        Returns:
            Structured response dict (see output_formatter.format_response).
        """
        t0 = time.perf_counter()

        # 1. Optional query expansion
        processed_query = query
        if self.query_processor:
            qp_result = self.query_processor.process(query)
            processed_query = qp_result["expanded"]

        # 2. Retrieve
        chunks = self._retrieve(processed_query, top_k, filters)

        if not chunks:
            return self._empty_result(query, t0)

        # 3. Generate
        llm_response = self._generate(query, chunks)

        # 4. Safety validation
        safety = self.guardrails.validate(
            response=llm_response.text,
            source_chunks=chunks,
        )

        # 5. Format
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return format_response(
            answer=safety.final_response,
            source_chunks=chunks,
            confidence=safety.confidence,
            grounding_score=safety.grounding_score,
            latency_ms=latency_ms,
            model=llm_response.model,
            is_safe=safety.is_safe,
            override_reason=safety.override_reason,
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks via the hybrid retriever or vector store."""
        if self.embedding_model is None:
            raise RuntimeError("embedding_model required for retrieval")

        query_embedding = self.embedding_model.embed_query(query)

        # HybridRetriever.retrieve() or VectorStore.search()
        if hasattr(self.retriever, "retrieve"):
            return self.retriever.retrieve(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
        elif hasattr(self.retriever, "search"):
            return self.retriever.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
        else:
            raise TypeError(f"Retriever {type(self.retriever)} has no retrieve() or search()")

    def _generate(self, query: str, chunks: list[dict[str, Any]]) -> LLMResponse:
        """Build prompt and call the LLM."""
        user_prompt = build_user_prompt(query, chunks)
        return self.llm.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def _empty_result(self, query: str, t0: float) -> dict[str, Any]:
        """Return a safe refusal when no chunks are retrieved."""
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return format_response(
            answer="No relevant sources were found for this query.",
            source_chunks=[],
            confidence=0.0,
            grounding_score=0.0,
            latency_ms=latency_ms,
            model=self.llm.model,
            is_safe=True,
            override_reason="no sources retrieved",
        )
