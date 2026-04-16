-- Clinical RAG initial schema. Runs automatically on first docker-compose up
-- (mounted into pgvector/pgvector:pg16's /docker-entrypoint-initdb.d/).
--
-- Application code (SQLAlchemy) will keep this schema in sync via Alembic
-- later, but we bootstrap pgvector + core tables here so a fresh clone can
-- run `docker compose up` and immediately connect.

CREATE EXTENSION IF NOT EXISTS vector;

-- Source documents (one row per article/guidance doc before chunking).
CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,
    source_type  TEXT NOT NULL,          -- 'pubmed', 'fda', 'clinical_trials', 'guideline'
    source_id    TEXT NOT NULL,          -- PMID, FDA docket, NCT ID, etc.
    title        TEXT,
    authors      TEXT,
    year         INT,
    doi          TEXT,
    url          TEXT,
    metadata     JSONB NOT NULL DEFAULT '{}',
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_type, source_id)
);

-- Retrieval chunks (one row per chunk; embedding column added after we know
-- the embedding dimension via SQLAlchemy migration — stub left here as NULL).
CREATE TABLE IF NOT EXISTS chunks (
    id            SERIAL PRIMARY KEY,
    document_id   INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,          -- position within the source document
    text          TEXT NOT NULL,
    token_count   INT,
    embedding     vector(768),           -- PubMedBERT default; re-create if model swaps
    metadata      JSONB NOT NULL DEFAULT '{}',
    UNIQUE (document_id, chunk_index)
);

-- Indexes
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source_type, year);

-- HNSW index on vector column for fast approximate nearest neighbor.
-- Only useful once embeddings are populated; safe to create empty.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Evaluation log (per-query metrics: latency, hallucination flag, confidence)
CREATE TABLE IF NOT EXISTS eval_log (
    id               SERIAL PRIMARY KEY,
    run_id           TEXT NOT NULL,          -- groups queries from one eval run
    question         TEXT NOT NULL,
    retrieved_ids    INT[] NOT NULL,         -- chunks.id array
    answer           TEXT,
    cited_sources    TEXT[],                 -- ['PMID: 123', 'FDA-2021-...']
    confidence       REAL,
    hallucination    BOOLEAN,
    latency_ms       INT,
    llm_provider     TEXT,
    llm_model        TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS eval_log_run_idx ON eval_log (run_id, created_at);
