# Quickstart -- Phase 1 Setup

This sets up the infrastructure: Postgres (with pgvector), Ollama, dependencies, and schema. You'll be able to run a health check that verifies everything is wired up correctly.

Tested on Linux + macOS. Expected time: 15-20 min (most is the Docker image pulls).

## Prerequisites

- Docker + Docker Compose
- Python 3.10+
- ~8 GB free disk (for Postgres + Ollama images + a small LLM)

## Steps

### 1. Clone + enter the repo

```bash
cd ~/workspaces/clinical-rag-assistant
```

### 2. Create a virtualenv and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
make dev-install
```

This installs the package in editable mode plus dev and eval extras (pytest, ruff, mypy, RAGAS).

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` -- at minimum change `POSTGRES_PASSWORD`. Leave cloud provider keys blank for now; Ollama is the default.

### 4. Start the infrastructure

```bash
make up
```

This starts two containers:

- `clinical-rag-postgres` -- PostgreSQL 16 with pgvector extension, auto-runs `scripts/init_db.sql` to create tables
- `clinical-rag-ollama` -- Ollama server on port 11434

The first run downloads a few GB of images. Subsequent runs are instant.

### 5. Pull the default Ollama model

```bash
make pull-ollama
```

Pulls BioMistral-7B (~4 GB). This is what the health check expects. You can switch to any Ollama model later by editing `LLM_MODEL` in `.env`.

### 6. Verify everything is wired up

```bash
make health
```

Expected output:

```
Clinical RAG Phase 1 health check
============================================================
  ✓ Postgres + pgvector: PostgreSQL 16.x ..., pgvector=0.7.x
  ✓ Schema (tables): tables present: chunks, documents, eval_log
  ✓ Config loader: llm.model=ollama/biomistral:7b, embedding=..., retrieval top_k=5
  ✓ LLM endpoint: ollama endpoint reachable (1 models pulled); ready
============================================================
4/4 checks passed
```

If any check fails, the error message tells you what to fix.

### 7. (Optional) Switch to Claude or another API LLM

Edit `.env`:

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=anthropic/claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...
```

Re-run `make health`. The LiteLLM client routes to whichever provider you set; no code changes needed.

Supported providers via LiteLLM: Ollama (default), Anthropic, OpenAI, Groq, Gemini, Azure OpenAI, and 100+ others. See https://docs.litellm.ai/docs/providers for full list.

## What's next

Phase 1 gives you: infrastructure running, schema in place, config loading, LLM reachable.

Phase 2 (next): PubMed loader targeting MTP-specific MeSH terms, FDA guidance scraper, chunker. You'll run `python -m ingest.pubmed_loader` to populate the `documents` and `chunks` tables.

## Troubleshooting

- **`make up` hangs on "Waiting for Postgres"** -- check `make logs`. Usually the first run is just slow to pull images.
- **`pgvector` extension not found** -- you're using a stock Postgres image instead of `pgvector/pgvector`. The docker-compose.yml already pins the right image; rebuild with `docker compose down -v && make up`.
- **LLM health check fails with "endpoint up, but model not pulled"** -- run `make pull-ollama`. If you changed `LLM_MODEL` in `.env`, pull that model instead: `docker exec clinical-rag-ollama ollama pull <model>`.
- **LiteLLM import error** -- `pip install -e ".[dev,eval]"` failed. Reinstall in a clean virtualenv.

## Common commands

```bash
make up         # start containers
make down       # stop containers
make logs       # tail docker logs
make health     # verify wiring
make test       # run pytest
make lint       # ruff + mypy
make format     # ruff format
make clean      # remove build artifacts
```
