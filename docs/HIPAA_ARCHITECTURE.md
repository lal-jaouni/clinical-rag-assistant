# HIPAA-Aware Architecture & Regulatory Compliance

## Overview

The Clinical RAG Assistant is designed as a **clinical decision support (CDS)** tool that retrieves published medical evidence and generates cited answers for healthcare professionals. This document describes the system's approach to HIPAA compliance, data handling, safety guarantees, deployment security, and regulatory positioning.

> **Important**: This system is not a medical device. It is a reference retrieval and summarization tool intended to support -- never replace -- clinical judgment.

---

## 1. Data Handling & PHI Protection

### 1.1 No PHI in the Data Pipeline

The system ingests only **publicly available** biomedical literature:

| Source | Data Type | PHI Risk |
|--------|-----------|----------|
| PubMed abstracts | Published research summaries | None -- public domain |
| ClinicalTrials.gov | Study protocols and results | None -- public registry |
| FDA guidance documents | Regulatory guidance PDFs | None -- public documents |

**No patient data enters the system at any point.** The vector store contains only chunked text from public sources, embeddings, and bibliographic metadata (PMID, trial ID, FDA document ID).

### 1.2 Query-Time Protections

User queries may inadvertently contain PHI (e.g., "My 45-year-old male patient with factor V Leiden..."). The system mitigates this through:

- **No query logging to persistent storage**: Queries are held in memory only for the duration of the request. The `/metrics` endpoint tracks aggregate statistics (counts, averages) but never stores query text.
- **No training on user input**: The system uses a frozen LLM via API call (Ollama). User queries are never used to fine-tune or update models.
- **Stateless inference**: Each request is independent. No session state, conversation history, or user context is retained between requests.
- **No external data transmission**: When using local Ollama models, all inference stays on-premises. No query data is sent to cloud LLM providers.

### 1.3 Audit Trail

For deployments requiring audit compliance (21 CFR Part 11):

- **Request metadata**: Timestamp, latency, confidence score, grounding score, safety override flag -- all tracked in-memory via the `/metrics` endpoint
- **No PII in logs**: Server logs contain only operational data (startup messages, error traces). Query content is excluded from log output.
- **Recommended additions for production**: Structured audit logging (JSON to file or SIEM), request IDs for traceability, user authentication tokens (not implemented in current version)

---

## 2. Safety Guarantees

### 2.1 Multi-Layer Hallucination Prevention

The system implements four independent safety mechanisms:

| Layer | Mechanism | Threshold | Action on Failure |
|-------|-----------|-----------|-------------------|
| 1. Source grounding | Embedding cosine similarity between answer sentences and retrieved chunks | 0.50 | Flag ungrounded sentences |
| 2. Confidence scoring | Composite of citation density, grounding score, and uncertainty language detection | 0.70 | Replace answer with safe refusal |
| 3. Hallucination detection | Sentence-level content-word overlap against source text | 0.70 | Flag hallucinated content |
| 4. Direct advice detection | Regex patterns for prescriptive language ("you should take", "administer X mg") | Any match | Override with safety disclaimer |

### 2.2 Safe Refusal Mechanism

When confidence falls below the threshold (default 0.70), the system replaces the generated answer with:

> "Insufficient confidence to provide a reliable answer. Please consult primary clinical sources or a healthcare provider."

This is a hard override -- the original answer is never returned to the user. The API response includes:
- `is_safe: false` flag
- `override_reason` explaining why the answer was replaced
- Original confidence and grounding scores for transparency

### 2.3 Source Attribution

Every answer includes numbered citations linking to specific source documents. Each source card contains:
- Citation label (e.g., `[PMID: 29451243]`)
- Direct URL to the source
- Publication year
- Relevance score from the retriever
- Text preview of the matching chunk

This allows clinicians to verify claims against primary literature before acting on any information.

---

## 3. Deployment Security

### 3.1 Network Architecture (Recommended Production)

```
                    ┌─────────────────────────────────────┐
                    │         Hospital Network / VPN       │
                    │                                      │
  Clinician ──TLS──►│  Reverse Proxy (nginx/Caddy)        │
                    │       │                              │
                    │       ▼                              │
                    │  FastAPI (port 8000, bind 127.0.0.1) │
                    │       │                              │
                    │       ├──► PostgreSQL + pgvector     │
                    │       │    (no external access)      │
                    │       │                              │
                    │       └──► Ollama LLM                │
                    │            (local inference only)    │
                    │                                      │
                    └─────────────────────────────────────┘
```

### 3.2 Security Checklist

| Control | Status | Notes |
|---------|--------|-------|
| TLS termination at reverse proxy | Recommended | Use Let's Encrypt or hospital CA |
| API authentication (API key or OAuth2) | Not implemented | Add before any multi-user deployment |
| RBAC (role-based access control) | Not implemented | Recommended for production |
| Database credentials in environment variables | Implemented | Via `.env` file, not in source code |
| No secrets in git history | Implemented | `.env` in `.gitignore` |
| Container isolation (Docker Compose) | Implemented | PostgreSQL runs in isolated container |
| Rate limiting | Not implemented | Add via reverse proxy or FastAPI middleware |
| Input validation | Implemented | Pydantic models enforce type/range constraints |
| SQL injection prevention | Implemented | SQLAlchemy ORM with parameterized queries |

### 3.3 Deployment Modes

**Development (current)**:
- Ollama running locally, HTTP only
- PostgreSQL in Docker with default credentials
- No authentication on API endpoints
- Suitable for: research, prototyping, demos

**Production (recommended)**:
- All components behind VPN or hospital firewall
- TLS on all connections
- API key or OAuth2 authentication
- Hardened PostgreSQL credentials
- Structured audit logging
- Suitable for: clinical pilot, institutional deployment

---

## 4. Regulatory Context

### 4.1 FDA Classification

Under the FDA's 2022 guidance on Clinical Decision Support Software, this system qualifies as a **non-device CDS** when it meets all four criteria:

1. **Not intended to acquire, process, or analyze medical images or signals** -- the system processes only published text
2. **Intended for healthcare professionals** -- designed for clinicians, not patients
3. **Intended to enable review of clinical evidence** -- retrieves and cites published sources
4. **Does not replace clinical judgment** -- provides information with citations, not diagnoses or treatment recommendations

> Reference: FDA Guidance, "Clinical Decision Support Software" (September 2022)

### 4.2 21 CFR Part 11 Considerations

For organizations subject to electronic records regulations:

| Requirement | Current State | Production Path |
|-------------|---------------|-----------------|
| Electronic signatures | Not applicable (no record creation) | Add if system generates clinical notes |
| Audit trails | Partial (in-memory metrics) | Add persistent structured logging |
| Access controls | Not implemented | Add authentication + RBAC |
| Data integrity | Ensured by PostgreSQL ACID | Add checksums on ingested documents |
| System validation | Test suite (164+ tests) | Add IQ/OQ/PQ documentation |

### 4.3 Related FDA Guidance Documents

- **Clinical Decision Support Software** (2022) -- primary classification guidance
- **Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD)** (2021) -- relevant if scope expands to diagnostic use
- **Good Machine Learning Practice for Medical Device Development** (2021) -- principles for ML system design
- **Predetermined Change Control Plans for ML-Enabled Device Software Functions** (2023) -- relevant for model update governance

---

## 5. Limitations & Disclaimers

### 5.1 System Limitations

- **Not a diagnostic tool**: The system retrieves and summarizes published evidence. It does not diagnose conditions, recommend treatments, or replace clinical expertise.
- **Knowledge currency**: The system's knowledge is limited to its ingested corpus. It does not have real-time access to the latest publications or drug safety alerts.
- **Domain scope**: Currently focused on trauma care, coagulation management, and related critical care topics. Questions outside this domain will receive lower confidence scores or refusals.
- **LLM limitations**: The underlying language model may occasionally generate plausible-sounding but incorrect text. The safety guardrails mitigate but cannot eliminate this risk entirely.
- **No drug interaction checking**: The system does not have access to drug databases and cannot verify medication interactions.

### 5.2 Intended Use Statement

This system is intended to assist healthcare professionals in reviewing published clinical evidence related to trauma and critical care. It is not intended to provide clinical recommendations, replace professional medical judgment, or serve as the sole basis for any clinical decision. All information should be verified against primary sources before clinical application.

### 5.3 Clinician Responsibility

Healthcare professionals using this system retain full responsibility for:
- Verifying retrieved information against primary sources
- Applying clinical judgment to patient-specific circumstances
- Recognizing the system's domain limitations
- Reporting any suspected errors or safety concerns

---

## 6. Compliance Roadmap

For organizations planning production deployment:

### Phase 1: Security Hardening
- [ ] Add API authentication (OAuth2 or API key)
- [ ] Configure TLS termination
- [ ] Restrict database access to application only
- [ ] Implement structured audit logging

### Phase 2: Validation
- [ ] Complete IQ/OQ/PQ documentation
- [ ] Document test coverage and safety thresholds
- [ ] Conduct clinical validation with domain experts
- [ ] Establish model update and revalidation procedures

### Phase 3: Monitoring
- [ ] Deploy monitoring dashboards (latency, confidence, safety overrides)
- [ ] Set up alerting for anomalous patterns
- [ ] Establish incident response procedures
- [ ] Schedule periodic re-evaluation of safety thresholds

---

*Last updated: 2026-04-17*
