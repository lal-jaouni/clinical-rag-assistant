# Phase 2a: PubMed Ingestion

Fetches PubMed abstracts via the NCBI Entrez API, chunks them, and writes documents + chunks to PostgreSQL. This is the minimum data layer the retriever (Phase 4) needs to run end-to-end; FDA guidance and ClinicalTrials.gov loaders (Phase 2b) land after.

## Design

Scope is deliberately narrow: **Massive Transfusion Protocol (MTP) decision support**. The 7 MeSH terms in `configs/ingest.yaml` cover the clinical concepts a trauma-bay resuscitation protocol actually invokes (hemorrhagic shock therapy, damage control resuscitation, tranexamic acid, blood products). Keeping the corpus narrow keeps retrieval precision high without a reranker.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| MeSH terms | 7 | Covers MTP decision path without dragging in unrelated emergency medicine |
| `max_per_term` | 100 | ~3-5s of Entrez traffic per term with the default rate limit |
| `date_range_start` | 2010 | Includes landmark trials (CRASH-2 2010, PROMMTT 2012, PROPPR 2015). In practice PubMed returns most-recent first so a 100-per-term cap returns 2023-2026 (see Known Limitations) |
| Chunk size | 200 tokens | Matches PubMedBERT's 512-token window with headroom for the query |
| Chunk overlap | 50 tokens | Preserves sentence-level context across boundaries |
| Chunk rule | Single chunk if abstract < 500 tokens, else greedy sentence-packing with overlap | Most PubMed abstracts are 150-400 tokens and fit in one chunk |

## Architecture

```
configs/ingest.yaml ─┐
                     ├─> config_loader.py ─> IngestConfig
.env (ENTREZ_*) ─────┘                             │
                                                   v
                                     scripts/ingest_pubmed.py
                                                   │
                          ┌────────────────────────┼────────────────────────┐
                          v                        v                        v
                    PubMedLoader             chunk_abstract              db/schema.py
                 (Entrez esearch +        (200-token chunks,            (Document,
                   efetch XML)              50-token overlap)             Chunk ORM)
                          │                        │                        │
                          └──> documents list ─────┴──> chunk dicts ────────┘
                                                   │
                                                   v
                                           PostgreSQL (pgvector)
                                        documents + chunks tables
                                        (embedding column = NULL; Phase 3 fills it)
```

## How to run

```bash
cd ~/workspaces/clinical-rag-assistant
make up                                     # Postgres + Ollama must be running
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/ingest_pubmed.py \
    2>&1 | tee logs/ingest_pubmed_$(date +%Y-%m-%d).log
```

`PYTHONUNBUFFERED=1` is required for `tee` to see progress in real time. The script is idempotent: re-running upserts existing PMIDs and replaces their chunks.

Expected output (15s on a warm cache):

```
Phase 1: Searching PubMed...
  Massive Transfusion: 100 results
  Hemorrhagic Shock/therapy: 100 results
  ... (5 more MeSH terms)
Found 625 unique PMIDs across 7 terms
Fetched 623 unique documents

Phase 2: Ingesting into database...
  [50/623] Committed 50 documents so far
  ...

======================================================================
INGESTION SUMMARY
======================================================================
MeSH terms queried:         7
Unique PMIDs (post-dedup):  623
Documents inserted:         623
Chunks inserted:            641
Parse errors:               0
Elapsed time:               14.4 seconds
======================================================================
```

## Verification

```bash
PGPASSWORD=$POSTGRES_PASSWORD psql -h localhost -U clinical_rag -d clinical_rag <<'SQL'
SELECT COUNT(*) AS documents FROM documents;
SELECT COUNT(*) AS chunks FROM chunks;
SELECT year, COUNT(*) FROM documents GROUP BY year ORDER BY year DESC;
SQL
```

## Gotchas encountered (and fixed)

Three issues stalled the first ingestion run. All are now handled in code, documented here so they stay fixed.

### 1. IPv6 DNS hang on NCBI

`eutils.ncbi.nlm.nih.gov` resolves AAAA records, but the VM's IPv6 egress silently blackholes traffic — `curl -6` hangs ~10s in `SYN-SENT` per request. biopython's `Entrez` library does not expose a per-request address family flag.

**Fix** (`src/ingest/pubmed_loader.py` top of file): monkeypatch `socket.getaddrinfo` at import time to force `AF_INET`:

```python
_original_getaddrinfo = socket.getaddrinfo
def _ipv4_only_getaddrinfo(host, port, family=0, *args, **kwargs):
    return _original_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)
socket.getaddrinfo = _ipv4_only_getaddrinfo
```

This is a process-wide monkeypatch; fine here because the loader is launched from a short-lived script. Revisit if the loader ever runs inside a long-lived service that also needs IPv6 for other hosts.

### 2. Entrez `read()` parses XML, not JSON

Early code passed `rettype="json"` to `Entrez.esearch` and then read `result["esearchresult"]["idlist"]`. `Entrez.read()` always returns a parsed XML tree with **capitalized top-level keys** (`Count`, `IdList`, `TranslationSet`). Passing `rettype="json"` silently returned XML that was then read as an empty dict, so every search returned 0 PMIDs.

**Fix**: drop `rettype="json"`; read `result["IdList"]`.

### 3. biopython returns `ListElement`, not `DictionaryElement`, for `AuthorList`

biopython 1.83 parses `<AuthorList>` into a `ListElement` (subclasses `list`), so the list of authors is the top-level object — not a dict with an `"Author"` key. Calling `.get("Author", [])` raises `AttributeError: 'ListElement' object has no attribute 'get'`. Same applies to `MeshHeadingList`, `PublicationTypeList`, and `ArticleIdList`.

**Fix** (`_extract_authors`): accept both shapes:

```python
if hasattr(author_list, "get"):
    authors = author_list.get("Author", None) or list(author_list)
else:
    authors = list(author_list)
```

The other extractors (`_extract_mesh_terms`, `_extract_publication_types`, `_extract_doi`) already iterate defensively (`isinstance(x, dict)` checks), so the ListElement path works for them without special casing.

## Known limitations

- **Corpus skews recent**: despite `date_range_start=2010`, the 100-per-term cap combined with PubMed's default "most recent first" ordering means the corpus currently spans 2023–2026. **Landmark MTP trials (CRASH-2 2010, PROMMTT 2012, PROPPR 2015) are not yet included.** Follow-up work: either raise `max_per_term` to 500, or add `sort="relevance"` to the esearch call, or hardcode a curated PMID list alongside the MeSH search. Tracked for Phase 2b polish.
- **Abstract-only**: no full-text ingestion. PubMed licensing prohibits redistribution of full text for most publishers. Fine for a portfolio demo; would need PMC OA subset or publisher-specific APIs for clinical production use.
- **Embeddings not generated yet**: the `chunks.embedding` column is `NULL` after this phase. Phase 3 (embed pipeline) fills it.
- **No delta ingestion**: re-running the script re-fetches all PMIDs; there is no "fetch only new since YYYY-MM-DD" mode. Not worth optimizing until corpus size justifies it.

## What's next

| Phase | Deliverable |
|-------|-------------|
| 2b | FDA SaMD PDF loader + ClinicalTrials.gov loader |
| 3  | PubMedBERT embedding pipeline → populate `chunks.embedding` |
| 4  | Hybrid retriever (pgvector + BM25 + RRF fusion) |

## Files in this phase

```
src/ingest/
  pubmed_loader.py      Entrez client with IPv4 monkeypatch + XML parsing
  chunker.py            Tokenizer-free sentence-level chunker
src/config_loader.py    IngestConfig: mesh_terms, max_per_term, date_range_start
configs/ingest.yaml     7 MTP MeSH terms, chunk params
scripts/ingest_pubmed.py  End-to-end runner with idempotent upserts
tests/test_chunker.py   14 tests for chunk_abstract
docs/PHASE_2A.md        This document
```
