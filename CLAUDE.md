# Clinical RAG Assistant - Project Instructions

## Project Overview
Clinical-grade RAG system for answering acute care questions with hallucination detection, source attribution, and HIPAA-aware architecture. Primary use case: acute blood product (ABP) transfusion protocols. Extensible to ED decision support and clinical documentation.

## Objectives (in priority order)
1. Build production-ready RAG pipeline with <2% hallucination rate on 50-100 clinical Q&A test set
2. Demonstrate clinical safety (confidence thresholding, "I don't know" handling, source attribution)
3. Create portfolio evidence for healthcare AI consulting ($150K-750K opportunities)
4. Minimize hallucinations and maximize retrieval precision (source precision >0.85)

## Tech Stack
- **Ingestion**: LlamaIndex (document loaders), Entrez API (PubMed), PDFPlumber (FDA docs), requests (ClinicalTrials.gov)
- **Embedding**: PubMedBERT or BioBERT (HuggingFace), sentence-transformers
- **Storage**: PostgreSQL + pgvector (self-hosted via Docker)
- **Retrieval**: pgvector (cosine/euclidean), Whoosh or rank-bm25 (hybrid), optional cross-encoder reranking
- **Generation**: Ollama (local inference) with Meditron-7B primary, BioMistral-7B fallback
- **Orchestration**: LangChain for chaining, LlamaIndex for retrieval
- **Evaluation**: RAGAS suite (faithfulness, answer_relevance, context_precision), custom hallucination detector
- **API**: FastAPI (endpoints), Streamlit (demo UI)
- **Testing**: pytest, mock data fixtures

## Code Organization

### Core Modules
- `src/ingest/` -- Document loaders, chunking pipeline
- `src/embed/` -- Embedding model selection, batch processing
- `src/retrieve/` -- Vector store, hybrid retrieval, reranking
- `src/generate/` -- LLM inference, safety guardrails, output formatting
- `src/evaluate/` -- RAGAS metrics, test Q&A set, hallucination tracking
- `src/api/` -- FastAPI endpoints, Streamlit UI
- `configs/` -- YAML configuration files (ingest, models, retrieval, evaluation)
- `data/` -- Document storage (raw, processed), test Q&A set
- `tests/` -- pytest suite with fixtures

### Key Classes & Interfaces
- `ingest.base.BaseDocumentLoader` -- Abstract loader, implement for new sources
- `ingest.pubmed_loader.PubMedLoader` -- Entrez client (IPv4-forced, XML-parsed). See `docs/PHASE_2A.md` for gotchas
- `ingest.chunker.chunk_abstract` -- Tokenizer-free function (not a class). 200-token chunks, 50-token overlap, single chunk if abstract <500 tokens
- `embed.models.EmbeddingModel` -- PubMedBERT/BioBERT wrapper
- `retrieve.vector_store.VectorStore` -- pgvector interface (search, metadata filter, add, delete)
- `retrieve.hybrid_retriever.HybridRetriever` -- Vector + BM25 fusion
- `generate.ollama_client.OllamaClient` -- Ollama integration
- `generate.safety_guardrails.SafetyGuardrails` -- Hallucination detection, confidence thresholding
- `evaluate.ragas_metrics.RAGASEvaluator` -- RAGAS integration

## Clinical & Safety Guidelines

### Hallucination Prevention
1. Confidence thresholding: if model confidence <0.7, return "I don't know" instead of uncertain answer
2. Source grounding: every fact in answer must map back to retrieved document chunk
3. Scope limiting: refuse queries outside medical literature (e.g., "treat my patient" -> redirect to provider)
4. Few-shot examples in prompts: ground model on clinical examples with proper citations

### Source Attribution
- Every answer includes citations with DOI or PubMed ID (e.g., "[PMID: 12345678]", "[FDA-2021-D-1234]")
- Include retrieval confidence score for transparency
- Link format: PubMed -> https://pubmed.ncbi.nlm.nih.gov/{pmid}, FDA -> https://www.fda.gov/...

### HIPAA Awareness
- No patient data stored locally (test Q&A set uses hypothetical scenarios only)
- No API keys or credentials in code (use environment variables)
- No logging of clinical data (log structure/queries, not clinical content)
- Document access controls (no git-tracking of raw clinical documents)

## Evaluation Metrics (target baselines)

| Metric | Target | Why |
|--------|--------|-----|
| Hallucination rate | <2% | Clinical safety non-negotiable |
| Source precision | >0.85 | Only cite sources actually supporting answer |
| Source recall | >0.70 | Retrieve all relevant sources for query |
| Answer latency | <3s | ED use case requires fast response |
| RAGAS faithfulness | >0.85 | Model answers supported by context |
| RAGAS relevance | >0.80 | Answers address clinical question directly |

## Testing Patterns

### Ingestion Tests
- Mock API responses (PubMed, ClinicalTrials.gov)
- Verify metadata extraction (DOI, date, source type)
- Test chunking with sample clinical abstracts
- Fixture: `sample_pubmed_response.json`, `sample_fda_pdf_path`

### Retrieval Tests
- Use LanceDB or SQLite for test vector DB (no Docker dependency)
- Verify hybrid fusion scoring (vector + BM25 weighting)
- Test metadata filters (date range, source type)
- Ground truth: hand-curated relevant chunks for clinical queries

### Generation Tests
- Mock Ollama responses (no inference needed in tests)
- Verify safety guardrails (confidence thresholding logic)
- Test prompt template rendering with clinical examples
- Verify citation formatting (PubMed links, confidence scores)

### Evaluation Tests
- RAGAS metrics on small ground truth set (5-10 examples)
- Hallucination detector on known hallucinating + correct responses
- End-to-end: query -> retrieval -> generation -> evaluation

## Configuration

All configs are YAML in `configs/`:

```yaml
# configs/ingest.yaml (Phase 2a -- narrow MTP scope)
pubmed:
  mesh_terms:
    - "Massive Transfusion"
    - "Hemorrhagic Shock/therapy"
    - "Resuscitation/methods"
    - "Tranexamic Acid/therapeutic use"
    - "Blood Transfusion/methods"
    - "Damage Control Resuscitation"
    - "Hemostatic Resuscitation"
  max_per_term: 100
  date_range_start: 2010
  cache_dir: "/tmp/pubmed_cache"

# configs/model.yaml
embedding:
  model: "pubmedbert-base-uncased-abstract"  # or "dmis-lab/biobert-v1.1"
  device: "cuda"  # or "cpu"

llm:
  provider: "ollama"
  model: "meditron:7b"  # primary
  fallback: "biomistral:7b"
  temperature: 0.2  # low temp for clinical safety
  max_tokens: 500

# configs/retrieval.yaml
chunking:
  chunk_size: 200  # tokens
  overlap: 50
  separator: "."  # sentence-level
  
retrieval:
  top_k: 5
  vector_weight: 0.6
  bm25_weight: 0.4
  rerank: true
  rerank_threshold: 0.5

# configs/evaluation.yaml
ragas:
  batch_size: 10
  metrics:
    - "faithfulness"
    - "answer_relevance"
    - "context_precision"

hallucination:
  confidence_threshold: 0.7
  track_per_source: true
  alert_threshold: 0.05  # flag if >5% hallucination
```

## Development Workflow

1. **Start**: Clone repo, install dependencies (`make dev-install`)
2. **Config**: Update `configs/` for your data sources and models
3. **Infrastructure**: `make up` to spin up Postgres + Ollama; `make health` to verify
4. **Ingest (Phase 2a, done)**: `PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/ingest_pubmed.py` loads the PubMed MTP corpus (~620 docs, ~640 chunks). Idempotent.
5. **Embed (Phase 3, next)**: Batch-embeds `chunks.text` with PubMedBERT; populates `chunks.embedding`
6. **Test**: `make test` (14 chunker tests green; retrieval/generate tests land with their phases)
7. **Evaluate (Phase 6)**: RAGAS suite on `data/qa_test_set.json`
8. **Deploy**: Push to GitHub, Docker images built via CI

## Deployment

### Local (Development)
```bash
docker-compose up -d
python -m src.api.app  # FastAPI on :8000
streamlit run src/api/streamlit_ui.py  # UI on :8501
```

### Production (if needed)
- Use managed PostgreSQL (AWS RDS, Azure Database)
- Deploy Ollama separately or use inference API
- FastAPI on Kubernetes or serverless (Cloud Run, Lambda)
- Monitor: latency, hallucination rate, source precision via logs

## Key Files to Know

- `README.md` -- Architecture, data sources, design decisions
- `CLAUDE.md` -- This file
- `pyproject.toml` -- Dependencies, build config
- `docker-compose.yaml` -- Infrastructure
- `configs/` -- Model and pipeline configuration
- `tests/` -- Test fixtures and ground truth data

## Avoiding Common Pitfalls

1. **Embedding model mismatch**: If embedding model doesn't match retrieval, vectors will be nonsensical. Lock embedding model in config.
2. **Chunking without context**: Chunking too small loses clinical context (e.g., dose in one chunk, indication in another). Use sentence-level with overlap.
3. **No source tracking**: If you don't store source metadata during ingestion, you can't attribute answers. Extract and preserve DOI, PubMed ID, date.
4. **Untuned thresholds**: Confidence threshold of 0.9 is too strict (high false-positive "I don't know"), 0.5 too loose (hallucinations leak through). Empirically tune on test set.
5. **Forgetting clinical context**: Prompts should include safety reminders ("You are a clinical research assistant. Cite sources. Never give medical advice to patients.").

## Questions or Blockers?

- Clinical domain: ask about terminology, safety guidelines, consulting angles
- Tech: check GitHub issues, README architecture section, tests for examples
- Evaluation: refer to RAGAS docs, test Q&A set in `data/qa_test_set.json`
