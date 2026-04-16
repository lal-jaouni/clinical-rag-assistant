"""Pydantic request/response models for API."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Clinical query request."""

    question: str = Field(..., description="Clinical question")
    domain_filter: Optional[str] = Field(None, description="Filter by medical domain")
    top_k: int = Field(5, description="Number of sources to retrieve")
    confidence_threshold: float = Field(0.7, description="Min confidence for answer")


class SourceCard(BaseModel):
    """Retrieved source document."""

    source_id: str = Field(..., description="PubMed ID, FDA doc ID, etc.")
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Chunk text")
    score: float = Field(..., description="Retrieval score (0-1)")
    url: str = Field(..., description="Link to full source")


class QueryResponse(BaseModel):
    """Clinical query response."""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceCard] = Field(..., description="Retrieved sources")
    confidence: float = Field(..., description="Answer confidence (0-1)")
    latency_ms: int = Field(..., description="Query latency")
    is_safe: bool = Field(..., description="Passed safety checks")
    override_reason: Optional[str] = Field(None, description="If answer was overridden by safety guardrails")
