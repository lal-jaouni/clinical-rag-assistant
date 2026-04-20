"""Pydantic request/response models for API."""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """Clinical query request."""

    question: str = Field(..., description="Clinical question")
    domain_filter: Optional[str] = Field(None, description="Filter by medical domain")
    top_k: int = Field(5, ge=1, le=20, description="Number of sources to retrieve")
    confidence_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Min confidence for answer"
    )


class SourceCard(BaseModel):
    """Retrieved source document."""

    index: int = Field(..., description="Citation index [1], [2], etc.")
    citation: str = Field(..., description="Citation label e.g. [PMID: 12345]")
    url: str = Field("", description="Link to full source")
    title: str = Field("", description="Document title")
    year: Optional[int] = Field(None, description="Publication year")
    source_type: str = Field("unknown", description="pubmed, fda, clinical_trials")
    source_id: str = Field("", description="PubMed ID, FDA doc ID, etc.")
    text_preview: str = Field("", description="Chunk text preview")
    relevance_score: Optional[float] = Field(None, description="Retrieval score")


class QueryResponse(BaseModel):
    """Clinical query response."""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceCard] = Field(default_factory=list, description="Retrieved sources")
    confidence: float = Field(..., description="Answer confidence (0-1)")
    grounding_score: float = Field(..., description="Source grounding score (0-1)")
    latency_ms: Optional[int] = Field(None, description="Query latency in ms")
    is_safe: bool = Field(..., description="Passed safety checks")
    override_reason: Optional[str] = Field(
        None, description="If answer was overridden by safety guardrails"
    )
    model: Optional[str] = Field(None, description="LLM model used")


class DataSourceInfo(BaseModel):
    """Metadata about a data source in the system."""

    source_type: str = Field(..., description="pubmed, fda, clinical_trials")
    count: int = Field(..., description="Number of documents")
    chunk_count: int = Field(..., description="Number of chunks")
    year_range: Optional[str] = Field(None, description="e.g. 2010-2024")


class MetricsResponse(BaseModel):
    """System-level RAG metrics."""

    total_queries: int = Field(0, description="Queries processed since startup")
    avg_latency_ms: float = Field(0.0, description="Average latency")
    p50_latency_ms: float = Field(0.0, description="p50 latency")
    p95_latency_ms: float = Field(0.0, description="p95 latency")
    hallucination_rate: float = Field(0.0, description="Hallucination rate (0-1)")
    avg_confidence: float = Field(0.0, description="Average confidence score")
    avg_grounding: float = Field(0.0, description="Average grounding score")
    safety_override_count: int = Field(0, description="Answers overridden by safety")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field("ok")
    model: Optional[str] = Field(None, description="Active LLM model")
    db_connected: bool = Field(False, description="Database connectivity")
    chunks_loaded: int = Field(0, description="Number of chunks in vector store")
