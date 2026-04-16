#!/usr/bin/env python3
"""Create GitHub issues for clinical-rag-assistant project."""

import subprocess
import json
from typing import List, Dict, Any

REPO = "lal-jaouni/clinical-rag-assistant"

ISSUES = [
    # Phase 1: Infrastructure
    {
        "title": "Phase 1.1: Set up Docker infrastructure (Postgres + pgvector + Ollama)",
        "body": """## Description
Set up Docker Compose configuration with Postgres, pgvector extension, and Ollama services for local development and testing.

## Acceptance Criteria
- docker-compose.yaml includes postgres:15 service with pgvector extension
- Ollama service configured with port 11434
- PostgreSQL service at port 5432 with automatic initialization
- Health checks for all services
- .env template for configuration
- Can run `docker-compose up -d` and all services healthy within 2 minutes

## Definition of Done
- docker-compose.yaml committed and tested locally
- README includes docker-compose setup instructions
- All services passing health checks
""",
        "labels": ["phase-1", "infrastructure"],
    },
    {
        "title": "Phase 1.2: Initialize PostgreSQL schema and pgvector setup",
        "body": """## Description
Create PostgreSQL schema for clinical documents, vector embeddings, and metadata. Initialize pgvector extension and create indexes.

## Acceptance Criteria
- documents table with columns: id, content, source, pmid/doi, publication_date, domain
- embeddings table with pgvector column (768-dim for PubMedBERT)
- metadata indexes on source, date, domain for efficient filtering
- Migration script for schema initialization
- Can run migrations from docker container
- Tests verify schema is queryable

## Definition of Done
- SQL migration files committed
- Schema documented in CLAUDE.md
- Seed data (3-5 sample documents) ingested successfully
""",
        "labels": ["phase-1", "infrastructure"],
    },
    {
        "title": "Phase 1.3: Set up project structure and configuration management",
        "body": """## Description
Finalize Python project structure, YAML config loaders, and environment variable handling.

## Acceptance Criteria
- pyproject.toml with all dependencies (llama-index, langchain, psycopg2-binary, pydantic, etc.)
- Config loaders for YAML files (ingest.yaml, model.yaml, retrieval.yaml, evaluation.yaml)
- Environment variable .env template
- utils.load_config() function used across modules
- pytest configuration (pyproject.toml or pytest.ini)
- All __init__.py files properly structured

## Definition of Done
- `pip install -e .` works without errors
- Config can be loaded and validated
- pytest runs and finds all tests
""",
        "labels": ["phase-1", "infrastructure"],
    },

    # Phase 2: Data Ingestion
    {
        "title": "Phase 2.1: Implement PubMed Entrez API loader",
        "body": """## Description
Build PubMed document loader using NCBI Entrez API. Fetch abstracts by MeSH terms, extract metadata (PMID, DOI, date), and handle batching and caching.

## Acceptance Criteria
- PubMedLoader class implements BaseDocumentLoader interface
- Supports batch fetching with configurable batch_size
- Extracts metadata: PMID, DOI, publication_date, authors, abstract
- Implements caching to avoid re-fetching
- Respects Entrez API rate limits (3 req/sec without key, 10 with)
- Handles errors gracefully (network, malformed responses)
- Unit tests with mock API responses
- Loads 10 sample abstracts from MeSH term "Hemorrhagic Shock"

## Definition of Done
- `python -m src.ingest.pubmed_loader --mesh "Hemorrhagic Shock" --limit 10` fetches documents
- Metadata extracted and validated
- Tests pass with mock data
""",
        "labels": ["phase-2", "documentation"],
    },
    {
        "title": "Phase 2.2: Implement FDA SaMD guidance document loader",
        "body": """## Description
Build FDA document loader for AI/ML SaMD guidance PDFs. Parse PDFs, extract text and metadata (date, document ID).

## Acceptance Criteria
- FDALoader class implements BaseDocumentLoader interface
- Fetches PDFs from FDA website (predefined URLs in config)
- Uses PDFPlumber or PyPDF2 for text extraction
- Extracts metadata: document_id, publication_date, document_type
- Implements local caching (configs/fda_cache or similar)
- Tests with 2-3 sample FDA guidance documents
- Handles corrupt/empty PDFs gracefully

## Definition of Done
- `python -m src.ingest.fda_loader` fetches and parses documents
- Metadata extracted and validated
- Tests pass
""",
        "labels": ["phase-2"],
    },
    {
        "title": "Phase 2.3: Implement ClinicalTrials.gov API loader",
        "body": """## Description
Fetch clinical trial information from ClinicalTrials.gov API. Extract protocol, inclusion/exclusion criteria, and trial metadata.

## Acceptance Criteria
- ClinicalTrialsLoader implements BaseDocumentLoader interface
- Fetches trials by condition/domain filter (Emergency Medicine, Transfusion, etc.)
- Extracts: trial_id, title, protocol_summary, inclusion_criteria, exclusion_criteria, status
- Filters by status (recruiting, active, completed)
- Implements caching
- Unit tests with mock API responses
- Loads 10 sample trials

## Definition of Done
- `python -m src.ingest.clinical_trials_loader --domain "Emergency Medicine"` fetches trials
- Tests pass with mock data
""",
        "labels": ["phase-2"],
    },
    {
        "title": "Phase 2.4: Implement clinical-aware chunking pipeline",
        "body": """## Description
Build chunking strategy optimized for clinical documents. Sentence-level splitting with overlap to preserve clinical context (e.g., diagnosis and treatment together).

## Acceptance Criteria
- ClinicalChunker class with configurable chunk_size (tokens), overlap, separator
- Preserves document metadata through chunking
- Sentence-level splitting (split on "." with clinical awareness)
- Handles edge cases: abbreviations (e.g., "Dr.", "i.e."), medical terminology
- Returns chunks with: text, metadata (source, pmid, chunk_id)
- Tests verify metadata preservation
- Tests verify overlap correctness

## Definition of Done
- Chunks sample PubMed abstract into 3-5 chunks with preserved metadata
- Tests pass
- Can configure chunk parameters via config
""",
        "labels": ["phase-2"],
    },

    # Phase 3: Embedding and Retrieval
    {
        "title": "Phase 3.1: Implement embedding model selection and inference",
        "body": """## Description
Select and implement PubMedBERT or BioBERT for clinical domain-specific embeddings. Create lazy-loading model wrapper for GPU/CPU inference.

## Acceptance Criteria
- EmbeddingModel class wraps sentence-transformers model
- Supports PubMedBERT ("pubmedbert-base-uncased-abstract") and BioBERT ("dmis-lab/biobert-v1.1")
- Lazy initialization (model loads only on first use)
- embed(texts) for batch, embed_query(query) for single
- Returns list of float vectors (768-dim for PubMedBERT)
- Tests verify embedding dimension and consistency
- Handles both CPU and CUDA devices

## Definition of Done
- Can embed a batch of clinical text snippets
- Embedding vectors have correct dimension
- Tests pass on CPU device
""",
        "labels": ["phase-3"],
    },
    {
        "title": "Phase 3.2: Implement batch embedding pipeline with pgvector storage",
        "body": """## Description
Create batch embedding pipeline with progress tracking and pgvector storage. Embed chunked documents and store in PostgreSQL with metadata.

## Acceptance Criteria
- batch_embed() function for efficient batching (configurable batch_size)
- Progress tracking (tqdm or similar)
- Stores embeddings to pgvector in PostgreSQL
- Validates embedding dimension matches model
- Handles large document sets (1K+ chunks)
- Tests verify end-to-end: chunk -> embed -> store
- Idempotent (re-running doesn't duplicate documents)

## Definition of Done
- 20 sample document chunks embedded and stored
- Can query by similarity from PostgreSQL
- Tests pass
""",
        "labels": ["phase-3", "infrastructure"],
    },
    {
        "title": "Phase 3.3: Implement vector store interface (pgvector queries)",
        "body": """## Description
Build VectorStore interface for semantic search, metadata filtering, and document management on pgvector.

## Acceptance Criteria
- VectorStore class with methods: search(), add_documents(), delete_documents()
- search(query_embedding, top_k, filters) returns scored results
- Supports metadata filters: source, date_range, domain
- Uses cosine or euclidean similarity (configurable)
- Returns results with: chunk_text, source_metadata, similarity_score
- Tests verify search accuracy and filtering
- Handles empty result sets gracefully

## Definition of Done
- Can search for similar documents to a query vector
- Metadata filtering works (e.g., source="pubmed", year>=2020)
- Tests pass
""",
        "labels": ["phase-3"],
    },
    {
        "title": "Phase 3.4: Implement hybrid retrieval (vector + BM25 fusion)",
        "body": """## Description
Combine vector similarity search and BM25 keyword matching for hybrid retrieval. Fuse results using reciprocal rank fusion (RRF) or weighted averaging.

## Acceptance Criteria
- HybridRetriever class takes VectorStore and BM25 index
- retrieve(query, query_embedding, top_k) returns fused results
- Configurable weights: vector_weight, bm25_weight
- Fusion method: RRF or weighted scoring
- Tests compare vector-only vs hybrid on terminology-heavy queries
- Handles case where BM25 has no results

## Definition of Done
- Retrieves clinical documents using both vector and keyword approaches
- Fused ranking makes sense (clinical terms prioritized)
- Tests pass
""",
        "labels": ["phase-3"],
    },
    {
        "title": "Phase 3.5: Implement cross-encoder reranking (optional)",
        "body": """## Description
Optional step: Add cross-encoder reranking to improve retrieval precision. Score (query, document) pairs directly to refine top-k results.

## Acceptance Criteria
- Reranker class uses sentence-transformers cross-encoder
- rerank(query, documents, top_k, threshold) returns filtered documents
- Improves precision on ground truth queries
- Optional toggle in config
- Tests show precision improvement (if available)

## Definition of Done
- Reranking improves retrieval precision
- Tests pass
- Optional disable without breaking retrieval
""",
        "labels": ["phase-3", "evaluation"],
    },

    # Phase 4: Generation and Safety
    {
        "title": "Phase 4.1: Implement Ollama LLM client for local inference",
        "body": """## Description
Build Ollama API client for local inference with Meditron-7B and BioMistral-7B models. Handle model fallback and error recovery.

## Acceptance Criteria
- OllamaClient class with generate(prompt, system_prompt) method
- Supports primary model (Meditron-7B) and fallback (BioMistral-7B)
- Configurable temperature (default 0.2 for clinical safety)
- Returns response dict with: text, model, confidence_estimate, raw_response
- is_available() checks Ollama server connectivity
- Graceful fallback if primary model unavailable
- Tests with mock responses (no real inference needed)

## Definition of Done
- OllamaClient can be initialized and configured
- generate() returns properly formatted response dict
- Tests pass
""",
        "labels": ["phase-4"],
    },
    {
        "title": "Phase 4.2: Implement clinical safety guardrails and hallucination detection",
        "body": """## Description
Build safety layer to detect and prevent hallucinations, direct medical advice, and out-of-scope responses.

## Acceptance Criteria
- SafetyGuardrails class with validate_response(response, sources, confidence)
- Confidence thresholding: if confidence < threshold, return "I don't know"
- detect_direct_medical_advice(text) returns True if response advises patient
- check_source_grounding(response, sources) scores evidence support (0-1)
- override_reason explains why answer was rejected
- Tests verify guardrails catch hallucinations and unsafe advice
- Target: <2% hallucination rate on test set

## Definition of Done
- SafetyGuardrails prevents low-confidence and unsafe responses
- Tests pass
- Metrics tracked for monitoring
""",
        "labels": ["phase-4", "evaluation"],
    },
    {
        "title": "Phase 4.3: Create clinical prompt templates with few-shot examples",
        "body": """## Description
Design clinical-specific prompt templates with system instructions and few-shot examples for safe, evidence-based responses.

## Acceptance Criteria
- PromptTemplates class with: system_prompt(), query_prompt(), few_shot_examples()
- System prompt emphasizes: cite sources, never give direct medical advice, say "I don't know"
- Few-shot examples (3-5) show proper citation format and scope boundaries
- query_prompt() formats question + context for LLM
- Tests verify templates compile without errors
- Templates reference config values (e.g., threshold, model name)

## Definition of Done
- Prompts load from PromptTemplates class
- Few-shot examples show desired behavior
- Tests pass
""",
        "labels": ["phase-4", "documentation"],
    },
    {
        "title": "Phase 4.4: Implement response formatting with source attribution",
        "body": """## Description
Format LLM responses with citations, confidence scores, and source cards for transparency and traceability.

## Acceptance Criteria
- OutputFormatter class with format_response(answer, sources, confidence) method
- Embeds citations: [PMID: 12345678], [FDA-2021-D-1234]
- Returns dict with: answer_text, sources (list of source cards), confidence
- SourceCard includes: source_id, title, snippet, score, url
- Formats PubMed links: https://pubmed.ncbi.nlm.nih.gov/{pmid}
- Formats FDA links appropriately
- Tests verify citation links are correctly formed

## Definition of Done
- Responses include properly formatted citations
- Source cards have all required metadata
- Tests pass
""",
        "labels": ["phase-4"],
    },

    # Phase 5: Evaluation
    {
        "title": "Phase 5.1: Create clinical Q&A test set (50-100 pairs)",
        "body": """## Description
Curate comprehensive test set of clinical questions with expected answers and source references for evaluation.

## Acceptance Criteria
- 50-100 Q&A pairs with coverage across: trauma, critical care, transfusion, emergency medicine
- Each pair includes: question, expected_answer_summary, source_ids (PMIDs/FDA docs), domain, difficulty
- Stored as JSON in data/qa_test_set.json
- Mixed difficulty: easy (lookup), medium (synthesis), hard (reasoning)
- At least 10 pairs per medical domain
- All sources are public (PubMed, FDA, ClinicalTrials.gov)
- ClinicalQASet class for loading and filtering by domain

## Definition of Done
- Q&A set has 50+ pairs in JSON format
- ClinicalQASet loads and filters correctly
- Tests pass
""",
        "labels": ["phase-3", "evaluation"],
    },
    {
        "title": "Phase 5.2: Implement RAGAS evaluation metrics",
        "body": """## Description
Integrate RAGAS (Retrieval-Augmented Generation Assessment) framework for standard RAG evaluation: faithfulness, answer_relevance, context_precision.

## Acceptance Criteria
- RAGASEvaluator class implements: faithfulness(), answer_relevance(), context_precision()
- Each metric returns score 0-1
- evaluate(questions, answers, contexts) returns dict of metric scores
- Target baselines: faithfulness >0.85, answer_relevance >0.80, context_precision >0.80
- Handles batches efficiently
- Tests verify metric behavior on known examples

## Definition of Done
- RAGAS metrics evaluate sample answers
- Scores match expected ranges
- Tests pass
""",
        "labels": ["phase-5", "evaluation"],
    },
    {
        "title": "Phase 5.3: Implement hallucination rate tracking and reporting",
        "body": """## Description
Build hallucination tracking pipeline to measure <2% target. Log hallucinations, generate reports, and integrate with CI/CD.

## Acceptance Criteria
- HallucinationDetector class with detect_hallucinations() and get_hallucination_rate()
- Log each evaluation with answer, sources, is_hallucinating
- Calculates: total hallucination count, percentage, per-source breakdown
- Alerts if rate exceeds threshold (default 5%)
- Reports saved to metrics/hallucination_report.json
- Tests verify detector on known hallucinating/grounded answers
- Target: <2% on test set

## Definition of Done
- Hallucination rate calculated across test set
- Reports generated with breakdown
- Tests pass
""",
        "labels": ["phase-5", "evaluation"],
    },
    {
        "title": "Phase 5.4: Set up evaluation pipeline and baseline metrics",
        "body": """## Description
Create end-to-end evaluation pipeline: ingest test Q&A, run through RAG, compute metrics, generate report.

## Acceptance Criteria
- Evaluation script: `python -m src.evaluate.pipeline --test-set data/qa_test_set.json`
- Runs RAG on all test questions
- Computes: RAGAS metrics, hallucination rate, latency, source precision/recall
- Generates report: metrics/evaluation_report.json with all scores
- Baseline metrics recorded (for regression testing)
- Tests verify pipeline runs without errors

## Definition of Done
- Evaluation completes successfully on test set
- Metrics report generated
- Baseline established for future comparisons
""",
        "labels": ["phase-5", "evaluation"],
    },

    # Phase 6: API and UI
    {
        "title": "Phase 6.1: Implement FastAPI endpoints for clinical RAG queries",
        "body": """## Description
Build FastAPI application with endpoints for querying, retrieving sources, and monitoring metrics.

## Acceptance Criteria
- POST /query with request: question, domain_filter, top_k, confidence_threshold
- Response includes: question, answer, sources (list), confidence, latency_ms, is_safe
- GET /sources returns available data sources and metadata
- GET /metrics returns system metrics (hallucination rate, latency_p50/p95, etc.)
- GET /health returns {"status": "ok"}
- Proper error handling and validation (Pydantic models)
- Tests verify endpoints with mock data

## Definition of Done
- FastAPI app starts without errors
- All endpoints respond correctly
- Tests pass
- API documentation auto-generated (Swagger)
""",
        "labels": ["phase-4"],
    },
    {
        "title": "Phase 6.2: Build Streamlit demo UI for clinical RAG",
        "body": """## Description
Create interactive Streamlit UI for querying the clinical RAG system with visualization of results and sources.

## Acceptance Criteria
- Input: clinical question, domain filter, confidence threshold slider
- Output: answer with citations, source cards with links, confidence score
- Visualizations: answer confidence bar, source relevance scores
- Sidebar: configuration options, metrics display
- Footer: disclaimer about verifying with current guidelines
- Responsive design (works on mobile)
- Tests verify Streamlit app loads without errors

## Definition of Done
- Streamlit app runs: `streamlit run src/api/streamlit_ui.py`
- Query interface functional
- Results displayed with sources
""",
        "labels": ["phase-6"],
    },

    # Documentation and Polish
    {
        "title": "Phase 7.1: Create HIPAA-aware architecture documentation",
        "body": """## Description
Document clinical and regulatory aspects: HIPAA compliance, data handling, safety guarantees, and deployment considerations.

## Acceptance Criteria
- Document in docs/HIPAA_ARCHITECTURE.md
- Covers: no PHI in logs, no data persistence beyond inference, audit trails
- Deployment security: network isolation, TLS for API, RBAC
- Explains hallucination prevention mechanisms
- Regulatory context: FDA oversight, clinical decision support guidance
- Limitations section: system not a substitute for clinical judgment
- References: 21 CFR Part 11, FDA guidance documents

## Definition of Done
- HIPAA document written and reviewed
- Deployment guide updated with security checklist
""",
        "labels": ["documentation"],
    },
    {
        "title": "Phase 7.2: Write feature catalog and API reference",
        "body": """## Description
Complete API documentation with endpoint specs, examples, and troubleshooting guide.

## Acceptance Criteria
- API_REFERENCE.md with all endpoints, request/response examples
- Feature catalog: ingestion sources, retrieval strategy, safety features
- Configuration guide: all YAML options explained
- Troubleshooting: common errors, Ollama/pgvector issues
- Performance tuning guide: chunk size, embedding batch size, retrieval top-k
- Security checklist for deployment

## Definition of Done
- API reference complete with examples
- All configuration options documented
- Troubleshooting covers common issues
""",
        "labels": ["documentation"],
    },
    {
        "title": "Phase 7.3: Write blog post on clinical RAG gaps and solution",
        "body": """## Description
Draft blog post highlighting the problem (open-source RAG lacks clinical safety), the solution (this project), and consulting applications.

## Acceptance Criteria
- Post structure: Problem -> Solution -> Results -> Consulting Implications
- Discusses: hallucination detection, source attribution, domain embeddings
- Includes code examples or architecture diagram
- Length: 1500-2500 words
- Highlights consulting use cases ($150K-750K opportunities)
- References research findings and GitHub repo
- Portfolio-ready draft (no sensitive info)

## Definition of Done
- Blog post draft complete
- Reviewed for clarity and technical accuracy
- Ready to publish on LinkedIn or personal blog
""",
        "labels": ["documentation"],
    },
]


def create_issue(issue_data: Dict[str, Any]) -> None:
    """Create a single GitHub issue."""
    title = issue_data["title"]
    body = issue_data["body"]
    labels = ",".join(issue_data.get("labels", []))

    cmd = [
        "gh",
        "issue",
        "create",
        "--repo",
        REPO,
        "--title",
        title,
        "--body",
        body,
    ]

    if labels:
        cmd.extend(["--label", labels])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Created: {title}")
        print(f"  {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {title}")
        print(f"  Error: {e.stderr}")


def main():
    """Create all issues."""
    print(f"Creating {len(ISSUES)} issues in {REPO}...\n")

    for issue in ISSUES:
        create_issue(issue)

    print(f"\nCompleted! {len(ISSUES)} issues created.")


if __name__ == "__main__":
    main()
