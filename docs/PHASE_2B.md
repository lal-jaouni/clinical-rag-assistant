# Phase 2b: ClinicalTrials.gov + FDA Ingestion

Phase 2b adds two more source types alongside the PubMed corpus from Phase 2a:

1. **ClinicalTrials.gov v2 API** — trial protocols for MTP-adjacent conditions (shipping in this branch)
2. **FDA SaMD guidance PDFs** — AI/ML regulatory guidance for the consulting angle (deferred to a follow-up PR)

This doc covers the CT.gov half. FDA PDF loader will land in a later commit and appendix.

## Why CT.gov alongside PubMed

PubMed abstracts give us the published evidence base; CT.gov gives us the **live protocol language** that clinical teams actually execute. The retriever will see both, so a query like *"when do you give cryoprecipitate in a massive transfusion protocol?"* can surface both the trial results (PubMed) and the inclusion/exclusion criteria, intervention arms, and dosing schemes (CT.gov). This is the pattern real ED decision-support systems use — review evidence + operational protocols, not one or the other.

## Scope

Same MTP focus as Phase 2a, but narrower because CT.gov tags are more heterogeneous than MeSH and broad queries pull in unrelated trials. The 5 condition strings in `configs/ingest.yaml::clinical_trials.conditions` match against CT.gov's conditions, keywords, and title fields (v2 `query.cond` semantics).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Conditions | 5 | `massive transfusion`, `hemorrhagic shock`, `damage control resuscitation`, `tranexamic acid trauma`, `trauma hemorrhage` |
| Statuses | COMPLETED, ACTIVE_NOT_RECRUITING, RECRUITING | Completed + active give proven protocols; recruiting gives forward-looking trials |
| `max_per_condition` | 100 | Paginated via `nextPageToken`; most CT.gov condition queries return well below this |
| `start_date_from` | 2010 | Excludes older, pre-damage-control-era trials |
| Chunk size / overlap | 200 / 50 | Same as PubMed, sharing one embedding model downstream |

## Architecture

```
configs/ingest.yaml ─┐
                     ├─> config_loader.py ─> IngestConfig (ct_* fields)
                     v
           scripts/ingest_clinical_trials.py
                        │
        ┌───────────────┼───────────────────────┐
        v               v                       v
ClinicalTrialsLoader   chunk_abstract      db/schema.py
  (v2 JSON API,    (shared w/ PubMed:     Document(source_type="clinical_trials")
   pageToken       200/50 tokens,          + Chunk rows
   pagination)      sentence-level)
        │               │                       │
        └──> study dicts ┴──> chunk dicts ──────┘
                            │
                            v
                 PostgreSQL (pgvector)
               (embedding = NULL; Phase 3 fills it)
```

The loader is source-type-aware: every Document row it writes has `source_type="clinical_trials"` and `source_id=nct_id`, reusing the same `(source_type, source_id)` unique constraint that keyed the PubMed rows. Retrieval in Phase 4 can mix or filter by source.

## Data shape

Each study is flattened from CT.gov's nested `protocolSection` into a flat dict before chunking:

| Field | Source | Notes |
|-------|--------|-------|
| `nct_id` | `identificationModule.nctId` | Primary key |
| `title` | `identificationModule.officialTitle` or `briefTitle` | Falls back to brief if official is missing |
| `summary` | `descriptionModule.briefSummary` | Chunked as main body |
| `detailed_description` | `descriptionModule.detailedDescription` | Appended to summary when present (dedup'd if identical) |
| `conditions` | `conditionsModule.conditions` | Stored in `meta` |
| `keywords` | `conditionsModule.keywords` | Stored in `meta` |
| `interventions` | `armsInterventionsModule.interventions` | Flattened to `"TYPE: name"` strings |
| `phases` | `designModule.phases` | Stored in `meta` |
| `status` | `statusModule.overallStatus` | Stored in `meta` |
| `enrollment` | `designModule.enrollmentInfo.count` | Stored in `meta` |
| `start_date` | `statusModule.startDateStruct.date` | `YYYY`, `YYYY-MM`, or `YYYY-MM-DD` |
| `year` | Parsed from `start_date` | Indexed on Document.year |
| `url` | `https://clinicaltrials.gov/study/{nct_id}` | Canonical source link |

Studies with no `briefSummary` and no `detailedDescription` are skipped — there is nothing to chunk.

## How to run

```bash
cd ~/workspaces/clinical-rag-assistant
make up                                     # Postgres must be running
mkdir -p logs
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/ingest_clinical_trials.py \
    2>&1 | tee logs/ingest_clinical_trials_$(date +%Y-%m-%d).log
```

Idempotent: re-running upserts the same NCT IDs and re-chunks them (existing chunks are deleted first, preventing duplicates).

Expected output:

```
Phase 1: Querying CT.gov v2...
  massive transfusion: ~40 fetched, ~40 new (total unique: 40)
  hemorrhagic shock: ~80 fetched, ~60 new (total unique: 100)
  ...

Phase 2: Ingesting into database...
  [50/~200] Committed 50 studies so far
  ...

CLINICAL TRIALS INGESTION SUMMARY
======================================================================
Conditions queried:         5
Unique trials (post-dedup): ~200
Documents inserted:         ~200
Chunks inserted:            ~400
Parse errors:               0
```

(Actual counts depend on CT.gov's current corpus and how it evolves; numbers above are order-of-magnitude expectations for the configured conditions.)

## Verification

```sql
SELECT source_type, COUNT(*) FROM documents GROUP BY source_type;
-- Expect: pubmed ~620, clinical_trials ~200

SELECT COUNT(*) AS chunks FROM chunks
 WHERE document_id IN (SELECT id FROM documents WHERE source_type = 'clinical_trials');

SELECT meta->>'status' AS status, COUNT(*)
  FROM documents
 WHERE source_type = 'clinical_trials'
 GROUP BY meta->>'status';
```

## Design choices

### Why v2 API (not v1)

CT.gov v1 was retired in 2024. v2 returns JSON directly (no XML parse), uses a sane module-per-protocol-section layout, and exposes `nextPageToken`-based pagination. No auth required for public searches.

### Why `query.cond` instead of advanced search expressions

CT.gov's advanced search supports a DSL, but `query.cond` does the right thing for condition-keyed searches (matches conditions, keywords, title), is easier to test, and matches the mental model of "give me trials about X" better than constructing boolean expressions. If later phases need precision controls, we can add a narrower `filter.advanced` expression without changing the loader's surface.

### Why client-side `start_date_from` filtering

The v2 API has no single "minimum start date" parameter equivalent to PubMed's date range. We could construct a `filter.advanced=AREA[StartDate]RANGE[2010-01-01,MAX]` expression, but per-condition filtering client-side is simpler, adds negligible cost (≤100 records per condition), and keeps the query shape obvious in logs.

### Why concatenate summary + detailed description

Many trials populate only `briefSummary`; others have a much richer `detailedDescription` with the actual protocol. Concatenating (summary first, then detailed when different) means the retriever sees the full protocol language without our having to pick one field and drop the other. Chunking handles length naturally.

## Known limitations

- **No full protocol PDF ingestion**: we only ingest the structured modules. CT.gov does not host full protocols for every trial, and we are not parsing the ones that are linked. Fine for decision support.
- **No outcome results**: `resultsSection.outcomeMeasures` is not ingested. Most MTP trials' primary outcome (24-hour all-cause mortality, massive transfusion incidence) is already captured in the brief summary / detailed description text; adding structured outcome measures would enable numeric retrieval but is out of scope for Phase 2b.
- **`nextPageToken` exhaustion**: if a condition really does exceed `max_per_condition=100` we stop paginating. For MTP this is well above what any single condition returns; revisit if broadening to general trauma/critical care.
- **Embeddings not populated**: Phase 3 responsibility. `chunks.embedding IS NULL` after this script finishes.

## FDA SaMD PDF loader — deferred

The original Phase 2b scope included FDA AI/ML guidance PDFs. That loader uses `pdfplumber` for text extraction + heuristic section parsing and a different storage key (`source_type="fda"`, `source_id=<FDA doc number>`). It ships in a follow-up PR; the `configs/ingest.yaml::fda` section already lists the target document types.

## Files in this phase

```
src/ingest/
  clinical_trials_loader.py     CT.gov v2 client (pagination, dedup, flatten)
src/config_loader.py            IngestConfig: ct_conditions, ct_statuses,
                                ct_max_per_condition, ct_start_date_from
configs/ingest.yaml             clinical_trials section
scripts/ingest_clinical_trials.py  End-to-end runner with idempotent upserts
tests/test_clinical_trials_loader.py  21 tests (parsing, pagination, dedup, filters)
docs/PHASE_2B.md                This document
```
