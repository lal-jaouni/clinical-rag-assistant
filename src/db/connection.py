"""Database connection: engine + session factory, reads settings from env."""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def _build_url() -> str:
    user = os.environ.get("POSTGRES_USER", "clinical_rag")
    password = os.environ.get("POSTGRES_PASSWORD", "changeme_localdev")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "clinical_rag")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a process-wide SQLAlchemy engine (cached)."""
    return create_engine(_build_url(), pool_pre_ping=True)


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session() -> Session:
    """Return a new SQLAlchemy session. Caller must close it."""
    return _session_factory()()
