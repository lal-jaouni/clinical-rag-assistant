"""Config loader: merges configs/*.yaml files with environment variable overrides.

Usage:
    from config_loader import load_config
    cfg = load_config()
    model_name = cfg.llm.model
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


# ── Typed sub-configs ────────────────────────────────────────────────────


class EmbeddingConfig(BaseModel):
    model: str = "pubmedbert-base-uncased-abstract"
    device: str = "cpu"
    batch_size: int = 32


class LLMConfig(BaseModel):
    # LiteLLM-format model string: "ollama/biomistral:7b", "anthropic/claude-3-5-sonnet-20241022", etc.
    # Env var LLM_MODEL overrides this.
    model: str = "ollama/biomistral:7b"
    temperature: float = 0.1
    max_tokens: int = 500
    top_p: float = 0.9
    # Extra provider args (e.g., api_base for Ollama)
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)


class SafetyConfig(BaseModel):
    confidence_threshold: float = 0.7
    source_grounding_threshold: float = 0.85
    max_hallucination_rate: float = 0.02


class ModelConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


class RetrievalConfig(BaseModel):
    vector_top_k: int = 5
    bm25_top_k: int = 3
    hybrid_enabled: bool = True
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    fusion_method: str = "rrf"
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranker_threshold: float = 0.5


class IngestConfig(BaseModel):
    mesh_terms: list[str] = Field(default_factory=list)
    max_per_term: int = 100
    date_range_start: int = 2010
    chunk_size: int = 200
    overlap: int = 50
    separator: str = "."
    entrez_email: str = ""
    entrez_api_key: str = ""


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["faithfulness", "answer_relevance", "context_precision"])
    hallucination_alert_threshold: float = 0.05
    latency_target_ms: int = 3000


class AppConfig(BaseModel):
    """Top-level config merged from all YAMLs + env overrides."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ── Loader ───────────────────────────────────────────────────────────────


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(cfg: AppConfig) -> AppConfig:
    """Pull selected env vars into the config tree.

    Kept explicit rather than magic -- each override documents what users
    can tweak without editing YAML.
    """
    if model := os.environ.get("LLM_MODEL"):
        cfg.model.llm.model = model
    if base := os.environ.get("OLLAMA_API_BASE"):
        if cfg.model.llm.model.startswith("ollama/"):
            cfg.model.llm.provider_kwargs.setdefault("api_base", base)
    if email := os.environ.get("ENTREZ_EMAIL"):
        cfg.ingest.entrez_email = email
    if key := os.environ.get("ENTREZ_API_KEY"):
        cfg.ingest.entrez_api_key = key
    return cfg


def load_config(config_dir: Path | None = None) -> AppConfig:
    """Merge configs/model.yaml, retrieval.yaml, ingest.yaml, evaluation.yaml.

    YAML files provide defaults; environment variables override specific keys.
    """
    cdir = config_dir or CONFIG_DIR

    raw: dict[str, Any] = {
        "model": _read_yaml(cdir / "model.yaml"),
        "retrieval": _read_yaml(cdir / "retrieval.yaml"),
        "ingest": _merge_ingest(_read_yaml(cdir / "ingest.yaml")),
        "evaluation": _read_yaml(cdir / "evaluation.yaml"),
    }

    cfg = AppConfig.model_validate(raw)
    return _apply_env_overrides(cfg)


def _merge_ingest(raw: dict[str, Any]) -> dict[str, Any]:
    """The ingest.yaml has pubmed.mesh_terms nested; flatten to IngestConfig fields."""
    pubmed = raw.get("pubmed", {})
    chunking = raw.get("chunking", {})
    return {
        "mesh_terms": pubmed.get("mesh_terms", []),
        "max_per_term": pubmed.get("max_per_term", 100),
        "date_range_start": pubmed.get("date_range_start", 2010),
        "chunk_size": chunking.get("chunk_size", 200),
        "overlap": chunking.get("overlap", 50),
        "separator": chunking.get("separator", "."),
    }
