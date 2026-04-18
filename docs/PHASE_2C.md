# Phase 2c: FDA SaMD Guidance Ingestion

## Goal
Add FDA AI/ML SaMD guidance documents to the clinical corpus so the assistant
can cite authoritative regulatory text alongside peer-reviewed literature and
active trials. This is the "regulatory retrieval" leg of the three-legged
corpus (PubMed + ClinicalTrials.gov + FDA).

## Sources
The manifest in `configs/ingest.yaml` targets five FDA publications covering
the current AI/ML SaMD regulatory stack:

| doc_id | Document | Date |
|--------|----------|------|
| `gmlp-2021` | Good Machine Learning Practice for Medical Device Development: Guiding Principles | 2021-10-27 |
| `aiml-samd-action-plan-2021` | AI/ML-Based SaMD Action Plan | 2021-01-12 |
| `pccp-final-2024` | Predetermined Change Control Plans for ML-Enabled Medical Devices (final guidance) | 2024-12-04 |
| `cds-final-2022` | Clinical Decision Support Software (final guidance) | 2022-09-28 |
| `samd-clinical-evaluation-2017` | Software as a Medical Device: Clinical Evaluation | 2017-12-08 |

Each manifest entry can also carry a `local_path` to bypass the download
entirely. This matters for air-gapped runs and for the case where FDA rotates
the `fda.gov/media/<n>/download` URLs after a publication is superseded.

## Pipeline
```
configs/ingest.yaml (fda.documents)
       │
       ▼
FDALoader.load()
  ├── _ensure_pdf  -> /tmp/fda_cache/<doc_id>.pdf  (cache-first download)
  ├── _extract_text -> pdfplumber per-page text, concatenated
  └── _parse_sections -> heuristic Roman-numeral header split
       │
       ▼
scripts/ingest_fda.py
  ├── _compose_text  -> "I. INTRODUCTION\n\nbody\n\nII. BACKGROUND\n\n..."
  ├── chunk_abstract -> 200-token sentence chunks with 50-token overlap
  └── upsert Document(source_type="fda", source_id=doc_id) + Chunks
```

## Section Parser
FDA guidance documents use consistent Roman-numeral section headers ("I.
INTRODUCTION", "II. BACKGROUND", ...). The parser uses a regex keyed on
all-caps body text to avoid catching mid-paragraph references like
"II. items". PDFs with no parseable structure fall back to a single `BODY`
section so callers can always rely on `doc["sections"]` being non-empty.

The section preamble (title page, TOC, issuance notices) is preserved as a
`PREAMBLE` pseudo-section rather than dropped -- that metadata is useful when
citing a specific guidance in the RAG response.

## Storage Contract
`documents` rows written by this phase:

| Column | Value |
|--------|-------|
| `source_type` | `"fda"` |
| `source_id` | stable `doc_id` from the manifest |
| `title` | guidance title |
| `year` | parsed from `issuance_date` |
| `url` | FDA.gov PDF URL (for citation) |
| `meta` | `{doc_type, issuance_date, pdf_path, section_count, section_names}` |

The `(source_type, source_id)` uniqueness constraint makes re-runs idempotent:
the runner deletes existing chunks and re-chunks rather than duplicating.

## Tests
`tests/test_fda_loader.py` runs fully offline via two monkeypatches:
- `requests.Session.get` returns canned PDF bytes per URL
- `pdfplumber.open` returns canned page text per filename

Coverage: year parsing, config validation, section parsing (with/without
headers, lowercase body guard), cache vs download semantics, end-to-end load
including failure-soft behavior on 404s and empty text.

## Known Limitations
- The Roman-numeral heuristic misses non-standard section formats (Appendix
  lettering, sub-bullets). Those end up concatenated into the parent section
  text, which is acceptable for retrieval-grade chunking but imperfect for
  citation precision.
- `pdfplumber` occasionally joins words across column breaks. For the demo
  corpus that's cosmetic; a future pass could swap in `unstructured` or
  `marker-pdf` for higher-fidelity layout extraction.
- Download failures fail-soft (skip with log) so a single 404 can silently
  shrink the corpus. The summary printout exposes fetched vs configured
  counts so the operator notices.

## Next Phases
- **Phase 3**: Embed all chunks (PubMed + CT.gov + FDA) with PubMedBERT and
  populate `chunks.embedding`.
- **Phase 4**: Retrieval -- hybrid BM25 + vector with RRF fusion, filtering
  by `source_type` when the query is obviously regulatory vs clinical.
- **Phase 5**: Answer generation with source grounding; cite FDA guidance by
  `doc_id` + section name, not just a PDF URL.
