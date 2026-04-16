"""FastAPI application for clinical RAG."""

from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from src.api.models import QueryRequest, QueryResponse


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        FastAPI app instance with routes for querying and metrics
    """
    app = FastAPI(
        title="Clinical RAG Assistant",
        description="Evidence-based clinical question answering",
        version="0.1.0",
    )

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        """Answer clinical question.

        Args:
            request: QueryRequest with question and filters

        Returns:
            QueryResponse with answer, sources, confidence
        """
        # Implementation placeholder
        pass

    @app.get("/sources")
    async def get_sources() -> List[Dict[str, Any]]:
        """List available data sources and their metadata."""
        # Implementation placeholder
        pass

    @app.get("/metrics")
    async def get_metrics() -> Dict[str, float]:
        """Get RAG system metrics (hallucination rate, latency, etc.)."""
        # Implementation placeholder
        pass

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Health check."""
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
