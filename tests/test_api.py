"""Tests for FastAPI endpoints using mocked pipeline components.

No database or LLM needed -- all dependencies are mocked.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app, _metrics_store


@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics store between tests."""
    _metrics_store["latencies"] = []
    _metrics_store["confidences"] = []
    _metrics_store["groundings"] = []
    _metrics_store["safety_overrides"] = 0
    _metrics_store["hallucination_flags"] = 0
    _metrics_store["total"] = 0


@pytest.fixture
def mock_pipeline():
    """Return a mock RAGPipeline that returns a realistic response."""
    pipeline = MagicMock()
    pipeline.llm.model = "ollama/llama3.1:8b"
    pipeline.answer.return_value = {
        "answer": "MTP is activated when >10 units pRBC in 24h [1].",
        "confidence": 0.85,
        "grounding_score": 0.72,
        "is_safe": True,
        "override_reason": None,
        "sources": [
            {
                "index": 1,
                "citation": "[PMID: 29451243]",
                "url": "https://pubmed.ncbi.nlm.nih.gov/29451243",
                "title": "MTP Guidelines",
                "year": 2020,
                "source_type": "pubmed",
                "source_id": "29451243",
                "text_preview": "Massive transfusion protocol is activated...",
                "relevance_score": 0.91,
            }
        ],
        "latency_ms": 1200,
        "model": "ollama/llama3.1:8b",
    }
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """TestClient with mocked pipeline injected (lifespan bypassed)."""
    import api.app as app_module

    # Bypass lifespan by creating app without it, then inject mocks
    from fastapi import FastAPI

    original_pipeline = app_module._pipeline
    original_session = app_module._session
    app_module._pipeline = mock_pipeline
    app_module._session = MagicMock()

    # Build app but patch lifespan to no-op
    with patch("api.app._build_pipeline"):
        app = create_app()
        # Override lifespan so TestClient doesn't try to connect
        app.router.lifespan_context = _noop_lifespan
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    app_module._pipeline = original_pipeline
    app_module._session = original_session


@pytest.fixture
def client_no_pipeline():
    """TestClient with no pipeline (simulates startup failure)."""
    import api.app as app_module

    original_pipeline = app_module._pipeline
    original_session = app_module._session
    app_module._pipeline = None
    app_module._session = None

    with patch("api.app._build_pipeline"):
        app = create_app()
        app.router.lifespan_context = _noop_lifespan
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    app_module._pipeline = original_pipeline
    app_module._session = original_session


# ------------------------------------------------------------------
# POST /query
# ------------------------------------------------------------------


class TestQueryEndpoint:
    def test_successful_query(self, client):
        resp = client.post("/query", json={"question": "What triggers MTP?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "What triggers MTP?"
        assert "MTP" in data["answer"]
        assert data["confidence"] == 0.85
        assert data["grounding_score"] == 0.72
        assert data["is_safe"] is True
        assert len(data["sources"]) == 1

    def test_query_with_custom_top_k(self, client, mock_pipeline):
        client.post("/query", json={"question": "test?", "top_k": 3})
        mock_pipeline.answer.assert_called_once_with(query="test?", top_k=3)

    def test_query_returns_source_cards(self, client):
        resp = client.post("/query", json={"question": "test?"})
        source = resp.json()["sources"][0]
        assert source["citation"] == "[PMID: 29451243]"
        assert source["source_type"] == "pubmed"
        assert source["year"] == 2020

    def test_low_confidence_overridden(self, client, mock_pipeline):
        mock_pipeline.answer.return_value["confidence"] = 0.3
        mock_pipeline.answer.return_value["override_reason"] = None
        resp = client.post(
            "/query",
            json={"question": "test?", "confidence_threshold": 0.7},
        )
        data = resp.json()
        assert "Insufficient confidence" in data["answer"]
        assert data["is_safe"] is False
        assert "confidence" in data["override_reason"]

    def test_pipeline_not_ready_returns_503(self, client_no_pipeline):
        resp = client_no_pipeline.post("/query", json={"question": "test?"})
        assert resp.status_code == 503

    def test_pipeline_error_returns_500(self, client, mock_pipeline):
        mock_pipeline.answer.side_effect = RuntimeError("LLM timeout")
        resp = client.post("/query", json={"question": "test?"})
        assert resp.status_code == 500
        assert "Pipeline error" in resp.json()["detail"]

    def test_missing_question_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_top_k_validation(self, client):
        resp = client.post("/query", json={"question": "test?", "top_k": 0})
        assert resp.status_code == 422
        resp = client.post("/query", json={"question": "test?", "top_k": 25})
        assert resp.status_code == 422


# ------------------------------------------------------------------
# GET /sources
# ------------------------------------------------------------------


class TestSourcesEndpoint:
    def test_returns_source_list(self, client):
        from sqlalchemy import text

        mock_session = client.app  # we need to reach the module-level _session
        import api.app as app_module

        app_module._session.execute.return_value.fetchall.return_value = [
            ("pubmed", 150, 620, 2010, 2024),
            ("fda", 8, 45, 2019, 2023),
            ("clinical_trials", 30, 90, 2015, 2024),
        ]

        resp = client.get("/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["source_type"] == "pubmed"
        assert data[0]["count"] == 150
        assert data[0]["chunk_count"] == 620

    def test_no_db_returns_503(self, client_no_pipeline):
        resp = client_no_pipeline.get("/sources")
        assert resp.status_code == 503


# ------------------------------------------------------------------
# GET /metrics
# ------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_empty_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_queries"] == 0

    def test_metrics_after_queries(self, client):
        # Fire some queries to populate metrics
        client.post("/query", json={"question": "q1?"})
        client.post("/query", json={"question": "q2?"})

        resp = client.get("/metrics")
        data = resp.json()
        assert data["total_queries"] == 2
        assert data["avg_latency_ms"] > 0
        assert data["avg_confidence"] > 0


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------


class TestHealthEndpoint:
    def test_healthy_system(self, client):
        import api.app as app_module

        app_module._session.execute.return_value.fetchone.return_value = (640,)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "ollama/llama3.1:8b"
        assert data["db_connected"] is True
        assert data["chunks_loaded"] == 640

    def test_degraded_no_pipeline(self, client_no_pipeline):
        resp = client_no_pipeline.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["db_connected"] is False


# ------------------------------------------------------------------
# OpenAPI docs
# ------------------------------------------------------------------


class TestDocs:
    def test_openapi_schema_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "/query" in schema["paths"]
        assert "/health" in schema["paths"]
        assert "/sources" in schema["paths"]
        assert "/metrics" in schema["paths"]
