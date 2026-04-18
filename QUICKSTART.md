# Quickstart -- Phases 1 + 2a Setup

This sets up the infrastructure (Postgres with pgvector, Ollama, dependencies, schema) and then ingests the PubMed MTP corpus so you have a populated database to query. You'll run a health check that verifies everything is wired up, then the ingestion script that loads ~620 abstracts.

Tested on Linux + macOS. Expected time: 15-20 min (most is the Docker image pulls), plus ~15s for ingestion.

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

Pulls Llama 3.1 8B (~4 GB). This is what the health check expects. You can switch to any Ollama model later by editing `LLM_MODEL` in `.env`.

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
  ✓ Config loader: llm.model=ollama/llama3.1:8b, embedding=..., retrieval top_k=5
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

### 8. Ingest the PubMed MTP corpus (Phase 2a)

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/ingest_pubmed.py \
    2>&1 | tee logs/ingest_pubmed_$(date +%Y-%m-%d).log
```

Fetches ~625 abstracts across 7 MTP MeSH terms (massive transfusion, hemorrhagic shock, damage control resuscitation, tranexamic acid, etc.), chunks them at 200 tokens with 50-token overlap, and writes to the `documents` and `chunks` tables. Idempotent — re-running upserts rather than duplicating.

Expected: `623 documents inserted, 641 chunks inserted` in ~15s. See `docs/PHASE_2A.md` for design notes, known limitations (corpus skews to 2023-2026), and the three gotchas that took a while to debug (IPv6 DNS hang on NCBI, Entrez XML vs JSON parsing, biopython ListElement shape).

Verify:

```bash
PGPASSWORD=$POSTGRES_PASSWORD psql -h localhost -U clinical_rag -d clinical_rag \
    -c "SELECT COUNT(*) FROM documents; SELECT COUNT(*) FROM chunks;"
```

## What's next

Phase 1 gave you: infrastructure running, schema in place, config loading, LLM reachable.
Phase 2a gives you: populated `documents` and `chunks` tables (abstracts + metadata, no embeddings yet).

Phase 2b (next): FDA SaMD guidance PDF loader and ClinicalTrials.gov loader. Same upsert pattern, different sources.
Phase 3: PubMedBERT embedding pipeline — fills the `embedding` column on `chunks` so vector search works.

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
