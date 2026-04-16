#!/usr/bin/env python3
"""Phase 2c: FDA SaMD guidance ingester.

Downloads (or re-uses cached) FDA AI/ML SaMD guidance PDFs, extracts text and
section structure, chunks, and upserts into PostgreSQL as documents with
source_type="fda". Idempotent: re-running with the same config re-chunks
existing fda_doc_ids rather than duplicating them.

Usage:
    source .venv/bin/activate
    python scripts/ingest_fda.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy.exc import IntegrityError

# Load env and add src to path
repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")
sys.path.insert(0, str(repo_root / "src"))

from config_loader import load_config
from db.connection import get_session
from db.schema import Chunk, Document
from ingest.chunker import chunk_abstract
from ingest.fda_loader import FDALoader


SOURCE_TYPE = "fda"


def _compose_text(doc: dict[str, Any]) -> str:
    """Build the text body we chunk and embed.

    Prefer section-structured text (header + body per section) when the loader
    detected Roman-numeral headers. Falls back to raw full-document text when
    the PDF had no parseable structure.
    """
    sections = doc.get("sections") or []
    if sections and not (len(sections) == 1 and sections[0]["name"] == "BODY"):
        parts = [f"{s['name']}\n\n{s['text']}" for s in sections if s.get("text")]
        return "\n\n".join(parts).strip()
    return (doc.get("text") or "").strip()


def ingest_fda() -> dict[str, Any]:
    """Main ingestion pipeline."""
    start_time = time.time()
    summary: dict[str, Any] = {
        "documents_configured": 0,
        "documents_fetched": 0,
        "documents_inserted": 0,
        "documents_updated": 0,
        "chunks_inserted": 0,
        "documents_skipped_no_text": 0,
        "parse_errors": [],
        "elapsed_seconds": 0,
    }

    cfg = load_config()
    if not cfg.ingest.fda_documents:
        print("ERROR: No fda.documents configured in ingest.yaml")
        return summary

    loader = FDALoader(
        documents=cfg.ingest.fda_documents,
        cache_dir=cfg.ingest.fda_cache_dir,
    )

    print("Ingesting FDA SaMD guidance PDFs:")
    print(f"  Documents configured: {len(cfg.ingest.fda_documents)}")
    print(f"  Cache dir:            {cfg.ingest.fda_cache_dir}")
    print(f"  Chunk size:           {cfg.ingest.chunk_size} tokens (overlap {cfg.ingest.overlap})")
    print()

    summary["documents_configured"] = len(cfg.ingest.fda_documents)

    print("Phase 1: Downloading + extracting...")
    documents = loader.load()
    summary["documents_fetched"] = len(documents)

    if not documents:
        print("No documents successfully fetched.")
        summary["elapsed_seconds"] = time.time() - start_time
        return summary

    print(f"Fetched {len(documents)} documents")
    print()

    print("Phase 2: Ingesting into database...")
    session = get_session()

    try:
        for doc_idx, doc_dict in enumerate(documents, 1):
            doc_id = doc_dict["fda_doc_id"]
            text = _compose_text(doc_dict)

            if not text:
                summary["documents_skipped_no_text"] += 1
                continue

            try:
                existing = (
                    session.query(Document)
                    .filter_by(source_type=SOURCE_TYPE, source_id=doc_id)
                    .first()
                )

                if existing:
                    existing.title = doc_dict.get("title", "")
                    existing.authors = ""  # FDA guidance has no author concept
                    existing.year = doc_dict.get("year")
                    existing.doi = ""
                    existing.url = doc_dict.get("url", "")
                    existing.meta = _meta(doc_dict)
                    session.query(Chunk).filter_by(document_id=existing.id).delete()
                    doc = existing
                    summary["documents_updated"] += 1
                else:
                    doc = Document(
                        source_type=SOURCE_TYPE,
                        source_id=doc_id,
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

                if doc_idx % 5 == 0:
                    session.commit()
                    print(f"  [{doc_idx}/{len(documents)}] Committed {doc_idx} documents so far")

            except IntegrityError as e:
                session.rollback()
                summary["parse_errors"].append(f"{doc_id}: {str(e)[:100]}")
                print(f"  [{doc_idx}] Integrity error for {doc_id}: {str(e)[:100]}")
            except Exception as e:
                session.rollback()
                summary["parse_errors"].append(
                    f"{doc_id}: {type(e).__name__}: {str(e)[:100]}"
                )
                print(
                    f"  [{doc_idx}] Error on {doc_id}: {type(e).__name__}: {str(e)[:100]}"
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
        "doc_type": doc.get("doc_type", ""),
        "issuance_date": doc.get("issuance_date", ""),
        "pdf_path": doc.get("pdf_path", ""),
        "section_count": len(doc.get("sections", [])),
        "section_names": [s["name"] for s in doc.get("sections", [])],
    }


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print("FDA INGESTION SUMMARY")
    print("=" * 70)
    print(f"Documents configured:        {summary['documents_configured']}")
    print(f"Documents fetched:           {summary['documents_fetched']}")
    print(f"Documents inserted:          {summary['documents_inserted']}")
    print(f"Documents updated:           {summary['documents_updated']}")
    print(f"Chunks inserted:             {summary['chunks_inserted']}")
    print(f"Documents skipped (no text): {summary['documents_skipped_no_text']}")
    print(f"Parse errors:                {len(summary['parse_errors'])}")
    if summary["parse_errors"]:
        for err in summary["parse_errors"][:5]:
            print(f"  - {err}")
        if len(summary["parse_errors"]) > 5:
            print(f"  ... and {len(summary['parse_errors']) - 5} more")
    print(f"Elapsed time:                {summary['elapsed_seconds']:.1f} seconds")
    print("=" * 70)


def verify_ingestion() -> None:
    """Query the DB to confirm FDA rows landed."""
    print()
    print("Verification:")
    print("-" * 70)
    session = get_session()
    try:
        fda_docs = session.query(Document).filter_by(source_type=SOURCE_TYPE).count()
        fda_chunks = (
            session.query(Chunk)
            .join(Document)
            .filter(Document.source_type == SOURCE_TYPE)
            .count()
        )

        print(f"FDA documents in DB: {fda_docs}")
        print(f"FDA chunks in DB:    {fda_chunks}")

        # doc_type distribution -- aggregated in Python to sidestep a Postgres
        # quirk where two separate CAST(metadata ->> 'doc_type' AS VARCHAR)
        # expressions emitted by SQLAlchemy's ORM are treated as non-equal in
        # the GROUP BY clause (GroupingError).
        meta_rows = (
            session.query(Document.meta)
            .filter(Document.source_type == SOURCE_TYPE)
            .all()
        )
        dtype_counts: dict[str, int] = {}
        for (meta,) in meta_rows:
            dtype = (meta or {}).get("doc_type") or ""
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        if dtype_counts:
            print()
            print("By doc_type:")
            for dtype, count in sorted(dtype_counts.items(), key=lambda kv: -kv[1]):
                print(f"  {dtype or '(blank)'}: {count}")

    finally:
        session.close()


def main() -> int:
    try:
        result = ingest_fda()
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
