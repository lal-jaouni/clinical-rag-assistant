# Clinical RAG Assistant -- API Reference

## Quick Start

```bash
cd clinical-rag-assistant
PYTHONPATH=src .venv/bin/python -m api.app                          # default model
PYTHONPATH=src .venv/bin/python -m api.app --model ollama/granite3.1-dense:2b
```

The server starts on `http://0.0.0.0:8000`. Interactive docs at `/docs` (Swagger) and `/redoc`.

---

## Endpoints

### POST /query

Answer a clinical question using the RAG pipeline.

**Request Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | *required* | Clinical question |
| `domain_filter` | string \| null | null | Filter by domain (trauma, coagulation, fda_regulatory, etc.) |
| `top_k` | int (1-20) | 5 | Number of source chunks to retrieve |
| `confidence_threshold` | float (0-1) | 0.7 | Minimum confidence; below this the answer is replaced with a safe refusal |

**Example Request:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What triggers activation of a massive transfusion protocol?",
    "top_k": 5,
    "confidence_threshold": 0.7
  }'
```

**Response (200):**

```json
{
  "question": "What triggers activation of a massive transfusion protocol?",
  "answer": "Massive transfusion protocol (MTP) is activated when a patient requires more than 10 units of packed red blood cells within 24 hours [1]. MTP activation is triggered in the setting of acute hemorrhage requiring rapid administration of blood products in specific ratios [1]...",
  "sources": [
    {
      "index": 1,
      "citation": "[PMID: 29451243]",
      "url": "https://pubmed.ncbi.nlm.nih.gov/29451243",
      "title": "Massive Transfusion Protocol Guidelines 2020",
      "year": 2020,
      "source_type": "pubmed",
      "source_id": "29451243",
      "text_preview": "Massive transfusion protocol is activated when a patient requires more than 10 units of packed red blood cells...",
      "relevance_score": 0.91
    }
  ],
  "confidence": 0.85,
  "grounding_score": 0.72,
  "latency_ms": 1200,
  "is_safe": true,
  "override_reason": null,
  "model": "ollama/llama3.1:8b"
}
```

**Error Responses:**
- `422`: Validation error (missing question, top_k out of range)
- `500`: Pipeline error (LLM timeout, retrieval failure)
- `503`: Pipeline not initialised (DB or model unavailable)

---

### GET /sources

List available data sources and their metadata.

**Example Request:**

```bash
curl http://localhost:8000/sources
```

**Response (200):**

```json
[
  {
    "source_type": "pubmed",
    "count": 150,
    "chunk_count": 620,
    "year_range": "2010-2024"
  },
  {
    "source_type": "clinical_trials",
    "count": 30,
    "chunk_count": 90,
    "year_range": "2015-2024"
  },
  {
    "source_type": "fda",
    "count": 8,
    "chunk_count": 45,
    "year_range": "2019-2023"
  }
]
```

---

### GET /metrics

Runtime metrics since server startup.

**Response (200):**

```json
{
  "total_queries": 47,
  "avg_latency_ms": 1850.3,
  "p50_latency_ms": 1200.0,
  "p95_latency_ms": 4500.0,
  "hallucination_rate": 0.0,
  "avg_confidence": 0.78,
  "avg_grounding": 0.65,
  "safety_override_count": 3
}
```

---

### GET /health

Component health check.

**Response (200):**

```json
{
  "status": "ok",
  "model": "ollama/llama3.1:8b",
  "db_connected": true,
  "chunks_loaded": 640
}
```

Status values: `"ok"` (all components ready) or `"degraded"` (pipeline or DB unavailable).

---

## Feature Catalog

### Ingestion Sources

| Source | Loader | Documents | Notes |
|--------|--------|-----------|-------|
| PubMed | `ingest.pubmed_loader.PubMedLoader` | ~620 abstracts | Entrez API, IPv4-forced, XML-parsed |
| ClinicalTrials.gov | `ingest.clinical_trials_loader.ClinicalTrialsLoader` | ~30 studies | REST API v2, paginated |
| FDA Guidance | `ingest.fda_loader.FDALoader` | ~8 documents | PDF extraction via PDFPlumber |

### Retrieval Strategy

1. **Query expansion**: Abbreviation expansion (MTP, TXA, DCR, etc.) via synonym map
2. **Embedding**: PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`) on CPU
3. **Hybrid search**: pgvector cosine similarity (weight 0.6) + BM25 keyword search (weight 0.4)
4. **Reciprocal Rank Fusion**: Merges vector and BM25 results with RRF scoring
5. **Top-K selection**: Default 5 chunks returned to the LLM

### Safety Features

| Feature | Implementation | Threshold |
|---------|---------------|-----------|
| Confidence scoring | Citation density + grounding + uncertainty detection | 0.7 |
| Source grounding | Embedding cosine similarity (with n-gram fallback) | 0.50 |
| Hallucination detection | Sentence-level content-word overlap against sources | 0.70 |
| Direct advice detection | Regex patterns for prescriptive language | Any match |
| Scope limiting | Out-of-domain detection via refusal | N/A |
| Safe refusal | Replaces low-confidence answers with refusal text | Automatic |

### Evaluation Metrics

| Metric | Method | Target |
|--------|--------|--------|
| Faithfulness | Token n-gram overlap between answer and sources | > 0.85 |
| Answer relevance | Token overlap between answer and question | > 0.80 |
| Context precision | Source coverage of ground truth | > 0.85 |
| Hallucination rate | Sentence-level grounding detection | < 2% |

---

## Configuration

### Environment Variables

Set in `.env` at project root:

```env
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpass
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=clinical_rag
OLLAMA_HOST=http://localhost:11434
```

### Server Options

```bash
PYTHONPATH=src .venv/bin/python -m api.app \
  --model ollama/llama3.1:8b \   # LLM model string
  --host 0.0.0.0 \               # Bind address
  --port 8000                     # Bind port
```

The model can also be set via `RAG_MODEL` environment variable.

### Pipeline Configuration

Defined in `configs/` YAML files:

| File | Controls |
|------|----------|
| `configs/ingest.yaml` | PubMed MeSH terms, date range, max per term |
| `configs/model.yaml` | Embedding model, LLM provider/model, temperature |
| `configs/retrieval.yaml` | Chunk size/overlap, top_k, vector/BM25 weights |
| `configs/evaluation.yaml` | RAGAS batch size, metric selection |

### Key Tuning Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `chunk_size` | retrieval.yaml | 200 tokens | Larger = more context per chunk, fewer chunks |
| `overlap` | retrieval.yaml | 50 tokens | Higher = better continuity, more storage |
| `vector_weight` | retrieval.yaml | 0.6 | Semantic vs keyword balance |
| `top_k` | /query request | 5 | More sources = richer context, slower |
| `confidence_threshold` | /query request | 0.7 | Higher = more refusals, fewer hallucinations |
| `temperature` | model.yaml | 0.1 | Lower = more deterministic, clinical use |

---

## Evaluation Pipeline

```bash
# Dry run (show test set stats)
PYTHONPATH=src .venv/bin/python -m evaluate.pipeline --dry-run

# Full eval with Llama 3.1 8B
PYTHONPATH=src .venv/bin/python -m evaluate.pipeline --model ollama/llama3.1:8b

# Eval subset (10 questions, stratified across domains)
PYTHONPATH=src .venv/bin/python -m evaluate.pipeline --limit 10

# Single domain
PYTHONPATH=src .venv/bin/python -m evaluate.pipeline --domain trauma

# Results saved to metrics/<model_slug>/
```

---

## Troubleshooting

### Ollama not responding

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull model if missing
ollama pull llama3.1:8b

# Check available models
ollama list
```

### PostgreSQL / pgvector issues

```bash
# Check Docker containers
docker compose ps

# Verify pgvector extension
docker exec -it clinical-rag-postgres psql -U raguser -d clinical_rag \
  -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# Check chunk count
docker exec -it clinical-rag-postgres psql -U raguser -d clinical_rag \
  -c "SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type;"
```

### Empty retrieval results

1. Verify chunks are embedded: `SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;`
2. Check BM25 index builds from DB: look for "Building BM25 index" in startup logs
3. Try a known query: "What triggers massive transfusion protocol?"

### High hallucination rate

1. Check grounding threshold (default 0.50) -- lower if answers are being falsely refused
2. Verify source chunks contain relevant text (not just metadata)
3. Consider increasing `top_k` to provide more context
4. Check if confidence threshold is too low (allowing uncertain answers through)

### Slow responses

1. CPU inference is expected to be 5-30s per query depending on model size
2. Granite 3.1 2B is ~3x faster than Llama 3.1 8B on CPU
3. Consider GPU acceleration for production: set `device: "cuda"` in model config
4. Reduce `top_k` to retrieve fewer chunks (less context = faster generation)
