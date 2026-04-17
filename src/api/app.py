"""FastAPI application for clinical RAG.

Usage:
    cd clinical-rag-assistant
    PYTHONPATH=src .venv/bin/python -m api.app                    # default (llama3.1:8b)
    PYTHONPATH=src .venv/bin/python -m api.app --model ollama/granite3.1-dense:2b
"""

from __future__ import annotations

import argparse
import bisect
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from api.models import (
    DataSourceInfo,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    SourceCard,
)

# ---------------------------------------------------------------------------
# Global state (initialised in lifespan)
# ---------------------------------------------------------------------------

_pipeline = None
_session = None
_metrics_store: dict[str, Any] = {
    "latencies": [],
    "confidences": [],
    "groundings": [],
    "safety_overrides": 0,
    "hallucination_flags": 0,
    "total": 0,
}


def _build_pipeline(model: str = "ollama/llama3.1:8b"):
    """Build the live RAG pipeline from environment variables."""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from embed.models import EmbeddingModel
    from generate.litellm_client import LLMClient
    from generate.rag_pipeline import RAGPipeline
    from generate.safety_guardrails import SafetyGuardrails
    from retrieve.hybrid_retriever import BM25Index, HybridRetriever
    from retrieve.query_processor import QueryProcessor
    from retrieve.vector_store import VectorStore

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

    db_url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}"
        f"/{os.environ['POSTGRES_DB']}"
    )
    engine = create_engine(db_url)
    session = sessionmaker(bind=engine)()

    embedding_model = EmbeddingModel(
        model_name="pubmedbert-base-uncased-abstract", device="cpu"
    )
    embedding_model.embed_query("warmup")

    llm = LLMClient(model=model, temperature=0.1, max_tokens=500)
    vector_store = VectorStore(session)
    bm25_index = BM25Index.from_db(session)
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        vector_weight=0.6,
        bm25_weight=0.4,
    )
    query_processor = QueryProcessor()
    guardrails = SafetyGuardrails(
        confidence_threshold=0.7,
        grounding_threshold=0.50,
        embedding_model=embedding_model,
    )

    pipeline = RAGPipeline(
        llm=llm,
        retriever=retriever,
        guardrails=guardrails,
        embedding_model=embedding_model,
        query_processor=query_processor,
    )
    return pipeline, session


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: build pipeline. Shutdown: close DB session."""
    global _pipeline, _session
    model = os.environ.get("RAG_MODEL", "ollama/llama3.1:8b")
    print(f"  Starting clinical RAG API (model={model})...")
    try:
        _pipeline, _session = _build_pipeline(model)
        print("  Pipeline ready.")
    except Exception as e:
        print(f"  WARNING: Pipeline init failed: {e}")
        print("  API will start but /query will return 503.")
    yield
    if _session:
        _session.close()
        print("  DB session closed.")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Clinical RAG Assistant",
        description=(
            "Evidence-based clinical question answering with source attribution, "
            "hallucination detection, and safety guardrails."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # POST /query
    # ------------------------------------------------------------------

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        """Answer a clinical question using the RAG pipeline.

        Retrieves relevant source chunks, generates an answer with citations,
        and validates via safety guardrails and hallucination detection.
        """
        if _pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not initialised. Check server logs.",
            )

        try:
            result = _pipeline.answer(
                query=request.question,
                top_k=request.top_k,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

        # Map source dicts to SourceCard models
        sources = [
            SourceCard(
                index=s.get("index", i + 1),
                citation=s.get("citation", ""),
                url=s.get("url", ""),
                title=s.get("title", ""),
                year=s.get("year"),
                source_type=s.get("source_type", "unknown"),
                source_id=s.get("source_id", ""),
                text_preview=s.get("text_preview", ""),
                relevance_score=s.get("relevance_score"),
            )
            for i, s in enumerate(result.get("sources", []))
        ]

        confidence = result.get("confidence", 0.0)
        grounding = result.get("grounding_score", 0.0)
        is_safe = result.get("is_safe", True)
        latency_ms = result.get("latency_ms")

        # Apply confidence threshold override
        answer = result.get("answer", "")
        override_reason = result.get("override_reason")
        if confidence < request.confidence_threshold and override_reason is None:
            answer = (
                "Insufficient confidence to provide a reliable answer. "
                "Please consult primary clinical sources or a healthcare provider."
            )
            override_reason = f"confidence {confidence:.3f} < threshold {request.confidence_threshold}"
            is_safe = False

        # Track metrics
        _metrics_store["total"] += 1
        _metrics_store["latencies"].append(latency_ms or 0)
        _metrics_store["confidences"].append(confidence)
        _metrics_store["groundings"].append(grounding)
        if not is_safe:
            _metrics_store["safety_overrides"] += 1

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            grounding_score=grounding,
            latency_ms=latency_ms,
            is_safe=is_safe,
            override_reason=override_reason,
            model=result.get("model"),
        )

    # ------------------------------------------------------------------
    # GET /sources
    # ------------------------------------------------------------------

    @app.get("/sources", response_model=list[DataSourceInfo])
    async def get_sources() -> list[DataSourceInfo]:
        """List available data sources and their metadata."""
        if _session is None:
            raise HTTPException(status_code=503, detail="Database not connected.")

        from sqlalchemy import text

        rows = _session.execute(
            text(
                "SELECT source_type, "
                "COUNT(DISTINCT source_id) AS doc_count, "
                "COUNT(*) AS chunk_count, "
                "MIN(year) AS min_year, "
                "MAX(year) AS max_year "
                "FROM chunks GROUP BY source_type ORDER BY source_type"
            )
        ).fetchall()

        return [
            DataSourceInfo(
                source_type=r[0] or "unknown",
                count=r[1],
                chunk_count=r[2],
                year_range=f"{r[3]}-{r[4]}" if r[3] and r[4] else None,
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # GET /metrics
    # ------------------------------------------------------------------

    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics() -> MetricsResponse:
        """Get system-level RAG metrics since server startup."""
        total = _metrics_store["total"]
        if total == 0:
            return MetricsResponse()

        lats = sorted(_metrics_store["latencies"])
        confs = _metrics_store["confidences"]
        grounds = _metrics_store["groundings"]

        return MetricsResponse(
            total_queries=total,
            avg_latency_ms=round(sum(lats) / len(lats), 1),
            p50_latency_ms=_percentile(lats, 50),
            p95_latency_ms=_percentile(lats, 95),
            hallucination_rate=round(
                _metrics_store["hallucination_flags"] / total, 4
            ),
            avg_confidence=round(sum(confs) / len(confs), 4),
            avg_grounding=round(sum(grounds) / len(grounds), 4),
            safety_override_count=_metrics_store["safety_overrides"],
        )

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check with component status."""
        db_ok = False
        chunk_count = 0
        if _session:
            try:
                from sqlalchemy import text

                row = _session.execute(text("SELECT COUNT(*) FROM chunks")).fetchone()
                chunk_count = row[0] if row else 0
                db_ok = True
            except Exception:
                pass

        return HealthResponse(
            status="ok" if _pipeline and db_ok else "degraded",
            model=_pipeline.llm.model if _pipeline else None,
            db_connected=db_ok,
            chunks_loaded=chunk_count,
        )

    return app


def _percentile(sorted_list: list[float], pct: int) -> float:
    """Compute percentile from a pre-sorted list."""
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * pct / 100)
    idx = min(idx, len(sorted_list) - 1)
    return round(sorted_list[idx], 1)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Clinical RAG API server")
    parser.add_argument(
        "--model",
        default="ollama/llama3.1:8b",
        help="LLM model (e.g., ollama/llama3.1:8b, ollama/granite3.1-dense:2b)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["RAG_MODEL"] = args.model

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
