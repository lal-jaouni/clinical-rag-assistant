#!/usr/bin/env python3
"""Phase 2b: ClinicalTrials.gov ingester.

Fetches study summaries via the CT.gov v2 JSON API, chunks the brief summary +
detailed description, and upserts into PostgreSQL. Idempotent: re-running with
the same config re-chunks existing NCT IDs rather than duplicating them.

Usage:
    source .venv/bin/activate
    python scripts/ingest_clinical_trials.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

# Load env and add src to path
repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")
sys.path.insert(0, str(repo_root / "src"))

from config_loader import load_config
from db.connection import get_session
from db.schema import Chunk, Document
from ingest.chunker import chunk_abstract
from ingest.clinical_trials_loader import ClinicalTrialsLoader


SOURCE_TYPE = "clinical_trials"


def _compose_text(doc: dict[str, Any]) -> str:
    """Build the text body we chunk and embed.

    CT.gov summaries are uneven: some trials have rich detailedDescription,
    others only briefSummary. Concatenate both (summary first, then detailed)
    so retrieval sees the full protocol language when available.
    """
    parts: list[str] = []
    summary = (doc.get("summary") or "").strip()
    detailed = (doc.get("detailed_description") or "").strip()
    if summary:
        parts.append(summary)
    if detailed and detailed != summary:
        parts.append(detailed)
    return "\n\n".join(parts).strip()


def ingest_clinical_trials() -> dict[str, Any]:
    """Main ingestion pipeline."""
    start_time = time.time()
    summary: dict[str, Any] = {
        "conditions_queried": 0,
        "unique_trials_after_dedup": 0,
        "documents_inserted": 0,
        "documents_updated": 0,
        "chunks_inserted": 0,
        "documents_skipped_no_text": 0,
        "parse_errors": [],
        "elapsed_seconds": 0,
    }

    cfg = load_config()
    if not cfg.ingest.ct_conditions:
        print("ERROR: No clinical_trials.conditions configured in ingest.yaml")
        return summary

    loader = ClinicalTrialsLoader(
        conditions=cfg.ingest.ct_conditions,
        statuses=cfg.ingest.ct_statuses,
        max_per_condition=cfg.ingest.ct_max_per_condition,
        start_date_from=cfg.ingest.ct_start_date_from,
    )

    print("Ingesting ClinicalTrials.gov studies:")
    print(f"  Conditions:        {len(cfg.ingest.ct_conditions)}")
    print(f"  Statuses:          {cfg.ingest.ct_statuses or 'ALL'}")
    print(f"  Max per condition: {cfg.ingest.ct_max_per_condition}")
    print(f"  Min start year:    {cfg.ingest.ct_start_date_from or 'no lower bound'}")
    print(f"  Chunk size:        {cfg.ingest.chunk_size} tokens (overlap {cfg.ingest.overlap})")
    print()

    print("Phase 1: Querying CT.gov v2...")
    documents = loader.load()
    summary["conditions_queried"] = len(cfg.ingest.ct_conditions)
    summary["unique_trials_after_dedup"] = len(documents)

    if not documents:
        print("No studies found.")
        summary["elapsed_seconds"] = time.time() - start_time
        return summary

    print(f"Fetched {len(documents)} unique studies")
    print()

    print("Phase 2: Ingesting into database...")
    session = get_session()

    try:
        for doc_idx, doc_dict in enumerate(documents, 1):
            nct_id = doc_dict["nct_id"]
            text = _compose_text(doc_dict)

            if not text:
                summary["documents_skipped_no_text"] += 1
                continue

            try:
                existing = (
                    session.query(Document)
                    .filter_by(source_type=SOURCE_TYPE, source_id=nct_id)
                    .first()
                )

                if existing:
                    existing.title = doc_dict.get("title", "")
                    existing.authors = ""  # CT.gov has no authors concept
                    existing.year = doc_dict.get("year")
                    existing.doi = ""
                    existing.meta = _meta(doc_dict)
                    session.query(Chunk).filter_by(document_id=existing.id).delete()
                    doc = existing
                    summary["documents_updated"] += 1
                else:
                    doc = Document(
                        source_type=SOURCE_TYPE,
                        source_id=nct_id,
                        title=doc_dict.get("title", ""),
                        authors="",
                        year=doc_dict.get("year"),
                        doi="",
                        url=doc_dict.get("url", ""),
                        meta=_meta(doc_dict),
                    )
                    session.add(doc)
                    session.flush()
                    summary["documents_inserted"] += 1

                chunks = chunk_abstract(
                    text,
                    chunk_size=cfg.ingest.chunk_size,
                    overlap=cfg.ingest.overlap,
                )

                for chunk_dict in chunks:
                    chunk = Chunk(
                        document_id=doc.id,
                        chunk_index=chunk_dict["chunk_index"],
                        text=chunk_dict["text"],
                        token_count=chunk_dict["token_count"],
                        embedding=None,  # Phase 3 populates embeddings
                        meta={
                            "chunk_index": chunk_dict["chunk_index"],
                            "total_chunks": len(chunks),
                        },
                    )
                    session.add(chunk)

                summary["chunks_inserted"] += len(chunks)

                if doc_idx % 50 == 0:
                    session.commit()
                    print(f"  [{doc_idx}/{len(documents)}] Committed {doc_idx} studies so far")

            except IntegrityError as e:
                session.rollback()
                summary["parse_errors"].append(f"{nct_id}: {str(e)[:100]}")
                print(f"  [{doc_idx}] Integrity error for {nct_id}: {str(e)[:100]}")
            except Exception as e:
                session.rollback()
                summary["parse_errors"].append(
                    f"{nct_id}: {type(e).__name__}: {str(e)[:100]}"
                )
                print(
                    f"  [{doc_idx}] Error on {nct_id}: {type(e).__name__}: {str(e)[:100]}"
                )

        session.commit()
        print("Committed final batch")

    finally:
        session.close()

    summary["elapsed_seconds"] = time.time() - start_time
    return summary


def _meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Assemble the JSON metadata blob stored on the Document row."""
    return {
        "conditions": doc.get("conditions", []),
        "keywords": doc.get("keywords", []),
        "interventions": doc.get("interventions", []),
        "phases": doc.get("phases", []),
        "status": doc.get("status", ""),
        "enrollment": doc.get("enrollment"),
        "start_date": doc.get("start_date", ""),
        "completion_date": doc.get("completion_date", ""),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print("CLINICAL TRIALS INGESTION SUMMARY")
    print("=" * 70)
    print(f"Conditions queried:         {summary['conditions_queried']}")
    print(f"Unique trials (post-dedup): {summary['unique_trials_after_dedup']}")
    print(f"Documents inserted:         {summary['documents_inserted']}")
    print(f"Documents updated:          {summary['documents_updated']}")
    print(f"Chunks inserted:            {summary['chunks_inserted']}")
    print(f"Documents skipped (no text):{summary['documents_skipped_no_text']}")
    print(f"Parse errors:               {len(summary['parse_errors'])}")
    if summary["parse_errors"]:
        for err in summary["parse_errors"][:5]:
            print(f"  - {err}")
        if len(summary["parse_errors"]) > 5:
            print(f"  ... and {len(summary['parse_errors']) - 5} more")
    print(f"Elapsed time:               {summary['elapsed_seconds']:.1f} seconds")
    print("=" * 70)


def verify_ingestion() -> None:
    """Query the DB to confirm CT.gov rows landed."""
    print()
    print("Verification:")
    print("-" * 70)
    session = get_session()
    try:
        ct_docs = (
            session.query(Document).filter_by(source_type=SOURCE_TYPE).count()
        )
        ct_chunks = (
            session.query(Chunk)
            .join(Document)
            .filter(Document.source_type == SOURCE_TYPE)
            .count()
        )

        print(f"CT.gov documents in DB: {ct_docs}")
        print(f"CT.gov chunks in DB:    {ct_chunks}")

        # Status distribution -- aggregated in Python to sidestep a Postgres
        # quirk where two separate CAST(metadata ->> 'status' AS VARCHAR)
        # expressions emitted by SQLAlchemy's ORM are treated as non-equal
        # in the GROUP BY clause (GroupingError). Fine for verification code.
        status_rows = (
            session.query(Document.meta)
            .filter(Document.source_type == SOURCE_TYPE)
            .all()
        )
        status_counts: dict[str, int] = {}
        for (meta,) in status_rows:
            status = (meta or {}).get("status") or ""
            status_counts[status] = status_counts.get(status, 0) + 1
        if status_counts:
            print()
            print("By overall status:")
            for status, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
                print(f"  {status or '(blank)'}: {count}")

        year_dist = (
            session.query(Document.year, func.count(Document.id))
            .filter(Document.source_type == SOURCE_TYPE, Document.year.isnot(None))
            .group_by(Document.year)
            .order_by(Document.year.desc())
            .limit(10)
            .all()
        )
        if year_dist:
            print()
            print("Top 10 years by start date:")
            for year, count in year_dist:
                print(f"  {year}: {count} studies")

    finally:
        session.close()


def main() -> int:
    try:
        result = ingest_clinical_trials()
        print_summary(result)
        verify_ingestion()
        return 0
    except Exception as e:
        print(f"FATAL ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
