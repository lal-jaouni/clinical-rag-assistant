# Clinical RAG Assistant

A retrieval-augmented generation (RAG) system for answering clinical questions over medical literature. Demonstrates LLM integration, vector search, and healthcare domain application.

## Why This Project

Healthcare AI roles increasingly require LLM/GenAI/RAG experience. This project demonstrates:
- **RAG architecture**: document ingestion, embedding, retrieval, generation
- **Healthcare domain**: medical literature, clinical terminology, safety-aware responses
- **Production patterns**: chunking strategies, evaluation metrics, Docker deployment
- **Open-source stack**: no proprietary API lock-in

## Target Job Skills Demonstrated

| Skill | How Demonstrated |
|-------|-----------------|
| LLM/GenAI | LangChain/LlamaIndex orchestration, prompt engineering |
| RAG | Vector search (pgvector/ChromaDB), document chunking, hybrid retrieval |
| Healthcare domain | PubMed abstracts, FDA guidance docs, clinical Q&A evaluation |
| Python | Clean library design, type hints, async patterns |
| Docker | Containerized deployment with docker-compose |
| PostgreSQL + pgvector | Vector similarity search at scale |
| Evaluation | Answer relevance, faithfulness, retrieval precision metrics (RAGAS) |

## Architecture

```
                    +------------------+
                    |   User Query     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Query Embedding  |
                    |  (sentence-transformers)
                    +--------+---------+
                             |
              +--------------v--------------+
              |    Vector Store (pgvector)   |
              |  PubMed abstracts + FDA docs |
              +--------------+--------------+
                             |
                    +--------v---------+
                    | Context Assembly  |
                    | (top-k retrieval) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   LLM Generation  |
                    | (Llama/Mistral    |
                    |  via Ollama)      |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Answer + Sources |
                    +------------------+
```

## Data Sources (all public, no PHI)

- **PubMed abstracts**: via Entrez API (NCBI), focused on trauma, critical care, hemodynamics
- **FDA guidance documents**: AI/ML SaMD guidance, 510(k) templates, PCCP framework
- **Clinical guidelines**: ATLS, hemorrhagic shock management protocols

## Project Structure

```
clinical-rag-assistant/
  src/
    ingest/           # Document loaders (PubMed, FDA, PDF)
    embed/            # Embedding models and chunking strategies
    retrieve/         # Vector search, hybrid retrieval, reranking
    generate/         # LLM integration, prompt templates, safety guardrails
    evaluate/         # RAGAS metrics, answer quality scoring
    api/              # FastAPI endpoints for query interface
  data/
    raw/              # Downloaded documents (gitignored)
    processed/        # Chunked and embedded documents (gitignored)
  configs/
    ingest.yaml       # Data source configs
    model.yaml        # LLM and embedding model configs
    retrieval.yaml    # Chunking, top-k, reranking params
  tests/
    test_ingest.py
    test_retrieval.py
    test_generation.py
    test_evaluation.py
  docker-compose.yaml # Postgres + pgvector + Ollama + app
  Dockerfile
  pyproject.toml
  README.md
```

## Implementation Plan

### Phase 1: Data Ingestion (2-3 hours)
- [ ] PubMed abstract fetcher (Entrez API, batch download by MeSH terms)
- [ ] FDA guidance PDF loader
- [ ] Chunking pipeline (sentence-level with overlap, metadata preservation)
- [ ] Store in pgvector with source metadata

### Phase 2: Retrieval Pipeline (2-3 hours)
- [ ] Embedding model selection (all-MiniLM-L6-v2 or clinical variant)
- [ ] Cosine similarity search via pgvector
- [ ] Hybrid retrieval: vector + keyword (BM25) fusion
- [ ] Source attribution in results

### Phase 3: Generation (2-3 hours)
- [ ] Ollama integration (Llama 3 or Mistral 7B)
- [ ] Prompt templates with clinical safety guardrails
- [ ] "I don't know" handling for out-of-scope queries
- [ ] Citation formatting (link back to PubMed/FDA source)

### Phase 4: Evaluation & Polish (2-3 hours)
- [ ] RAGAS evaluation suite (faithfulness, relevance, context precision)
- [ ] Example query set with expected answers
- [ ] FastAPI endpoint for demo
- [ ] Docker-compose for one-command setup
- [ ] README with demo GIF

## Key Design Decisions

1. **Ollama over API**: demonstrates self-hosted LLM capability, no API costs, runs on local GPU
2. **pgvector over Pinecone**: stays in PostgreSQL ecosystem (matches job requirements), self-hosted
3. **Clinical safety guardrails**: model should never give direct medical advice, always cite sources
4. **Public data only**: no PHI, no NDA-restricted content, fully open-sourceable when ready

## License

MIT (will be made public when portfolio-ready)
