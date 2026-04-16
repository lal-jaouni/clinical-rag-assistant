#!/usr/bin/env python3
"""Phase 1 health check: verify docker services are up and configured correctly.

Run after `docker compose up -d`. Exits 0 on success, non-zero on failure.
Designed so a fresh clone can verify "is everything wired up right?" in one command.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env if it exists (so POSTGRES_PASSWORD, ANTHROPIC_API_KEY, etc. are available).
repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

# Put src/ on the path so we can import from the package
sys.path.insert(0, str(repo_root / "src"))


def check(label: str, fn):
    """Run a check, print PASS/FAIL and exit code on failure."""
    try:
        detail = fn()
        print(f"  \u2713 {label}: {detail}")
        return True
    except Exception as e:
        print(f"  \u2717 {label}: {type(e).__name__}: {e}")
        return False


def check_postgres() -> str:
    from sqlalchemy import text
    from db.connection import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar_one()
        pgvector = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname='vector'")
        ).scalar()
        if not pgvector:
            raise RuntimeError("pgvector extension not installed in DB")
    short_ver = version.split(",")[0]
    return f"{short_ver}, pgvector={pgvector}"


def check_tables() -> str:
    from sqlalchemy import inspect
    from db.connection import get_engine

    insp = inspect(get_engine())
    tables = set(insp.get_table_names())
    required = {"documents", "chunks", "eval_log"}
    missing = required - tables
    if missing:
        raise RuntimeError(f"missing tables: {missing}. Did init_db.sql run?")
    return f"tables present: {', '.join(sorted(required))}"


def check_config() -> str:
    from config_loader import load_config

    cfg = load_config()
    return (
        f"llm.model={cfg.model.llm.model}, "
        f"embedding={cfg.model.embedding.model}, "
        f"retrieval top_k={cfg.retrieval.vector_top_k}"
    )


def check_llm() -> str:
    """Try to reach the configured LLM. Does NOT fail if Ollama model isn't
    pulled yet — just reports whether the endpoint is reachable."""
    from config_loader import load_config
    from generate.litellm_client import LLMClient

    cfg = load_config()
    client = LLMClient.from_config(cfg.model.llm)

    # For Ollama we just check the endpoint; pulling the model is a manual step.
    if cfg.model.llm.model.startswith("ollama/"):
        import requests

        base = cfg.model.llm.provider_kwargs.get("api_base") or os.environ.get(
            "OLLAMA_API_BASE", "http://localhost:11434"
        )
        r = requests.get(f"{base}/api/tags", timeout=5)
        r.raise_for_status()
        tags = r.json().get("models", [])
        names = [t.get("name") for t in tags]
        model_name = cfg.model.llm.model.split("/", 1)[1]
        pulled = any(n == model_name or n.startswith(model_name.split(":")[0]) for n in names)
        status = "ready" if pulled else f"endpoint up, but '{model_name}' not pulled"
        return f"ollama endpoint reachable ({len(names)} models pulled); {status}"

    # For cloud providers, just confirm the API key is set; don't spend tokens.
    provider = client.provider
    key_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "azure": "AZURE_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }.get(provider)
    if key_var and not os.environ.get(key_var):
        raise RuntimeError(f"{key_var} not set; add to .env")
    return f"provider={provider}, api key set"


def main() -> int:
    print("Clinical RAG Phase 1 health check")
    print("=" * 60)
    checks = [
        ("Postgres + pgvector", check_postgres),
        ("Schema (tables)", check_tables),
        ("Config loader", check_config),
        ("LLM endpoint", check_llm),
    ]
    results = [check(label, fn) for label, fn in checks]
    print("=" * 60)
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
