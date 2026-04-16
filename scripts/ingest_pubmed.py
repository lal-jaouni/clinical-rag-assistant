#!/usr/bin/env python3
"""Phase 2a: PubMed MTP literature ingester.

Fetches abstracts via NCBI Entrez API, chunks them, and stores in PostgreSQL.
Idempotent: re-running with same config upserts existing documents and re-chunks.

Usage:
    source .venv/bin/activate
    python scripts/ingest_pubmed.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import delete, text
from sqlalchemy.exc import IntegrityError

# Load env and add src to path
repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")
sys.path.insert(0, str(repo_root / "src"))

from config_loader import load_config
from db.connection import get_session
from db.schema import Document, Chunk
from ingest.pubmed_loader import PubMedLoader
from ingest.chunker import chunk_abstract


def ingest_pubmed() -> dict[str, Any]:
    """Main ingestion pipeline.

    Returns:
        Summary dict with counts and timing
    """
    start_time = time.time()
    summary = {
        "mesh_terms_queried": 0,
        "total_pmids_before_dedup": 0,
        "unique_pmids_after_dedup": 0,
        "documents_inserted": 0,
        "documents_updated": 0,
        "chunks_inserted": 0,
        "documents_skipped_no_abstract": 0,
        "parse_errors": [],
        "elapsed_seconds": 0,
    }

    # Load config
    cfg = load_config()
    if not cfg.ingest.mesh_terms:
        print("ERROR: No MeSH terms configured in ingest.yaml")
        return summary

    # Initialize loader
    loader = PubMedLoader(
        api_key=cfg.ingest.entrez_api_key,
        email=cfg.ingest.entrez_email,
        mesh_terms=cfg.ingest.mesh_terms,
        max_per_term=getattr(cfg.ingest, "max_per_term", 100),
        date_range_start=getattr(cfg.ingest, "date_range_start", 2010),
    )

    print(f"Ingesting PubMed literature:")
    print(f"  MeSH terms: {len(cfg.ingest.mesh_terms)} terms")
    print(f"  Chunk size: {cfg.ingest.chunk_size} tokens")
    print(f"  Overlap: {cfg.ingest.overlap} tokens")
    print()

    # Fetch documents
    print("Phase 1: Searching PubMed...")
    documents = loader.load()
    summary["unique_pmids_after_dedup"] = len(documents)
    summary["mesh_terms_queried"] = len(cfg.ingest.mesh_terms)

    if not documents:
        print("No documents found.")
        summary["elapsed_seconds"] = time.time() - start_time
        return summary

    print(f"Fetched {len(documents)} unique documents")
    print()

    # Ingest into database
    print("Phase 2: Ingesting into database...")
    session = get_session()

    try:
        for doc_idx, doc_dict in enumerate(documents, 1):
            pmid = doc_dict["pmid"]
            abstract = doc_dict.get("abstract", "")

            # Skip if no abstract
            if not abstract or not abstract.strip():
                summary["documents_skipped_no_abstract"] += 1
                if doc_idx % 50 == 0:
                    print(f"  [{doc_idx}/{len(documents)}] Skipped {pmid} (no abstract)")
                continue

            try:
                # Upsert document row
                existing = (
                    session.query(Document)
                    .filter_by(source_type="pubmed", source_id=pmid)
                    .first()
                )

                if existing:
                    # Update metadata and delete old chunks
                    existing.title = doc_dict.get("title", "")
                    existing.authors = doc_dict.get("authors", "")
                    existing.year = doc_dict.get("year")
                    existing.doi = doc_dict.get("doi", "")
                    existing.meta = {
                        "mesh_terms": doc_dict.get("mesh_terms", []),
                        "publication_types": doc_dict.get("publication_types", []),
                    }
                    session.query(Chunk).filter_by(document_id=existing.id).delete()
                    doc = existing
                    summary["documents_updated"] += 1
                else:
                    # Create new document
                    doc = Document(
                        source_type="pubmed",
                        source_id=pmid,
                        title=doc_dict.get("title", ""),
                        authors=doc_dict.get("authors", ""),
                        year=doc_dict.get("year"),
                        doi=doc_dict.get("doi", ""),
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        meta={
                            "mesh_terms": doc_dict.get("mesh_terms", []),
                            "publication_types": doc_dict.get("publication_types", []),
                        },
                    )
                    session.add(doc)
                    session.flush()  # Get the auto-generated ID
                    summary["documents_inserted"] += 1

                # Chunk the abstract
                chunks = chunk_abstract(
                    abstract,
                    chunk_size=cfg.ingest.chunk_size,
                    overlap=cfg.ingest.overlap,
                )

                for chunk_dict in chunks:
                    chunk = Chunk(
                        document_id=doc.id,
                        chunk_index=chunk_dict["chunk_index"],
                        text=chunk_dict["text"],
                        token_count=chunk_dict["token_count"],
                        embedding=None,  # Phase 3: embeddings
                        meta={
                            "chunk_index": chunk_dict["chunk_index"],
                            "total_chunks": len(chunks),
                        },
                    )
                    session.add(chunk)

                summary["chunks_inserted"] += len(chunks)

                if doc_idx % 50 == 0:
                    session.commit()
                    print(f"  [{doc_idx}/{len(documents)}] Committed {doc_idx} documents so far")

            except IntegrityError as e:
                session.rollback()
                summary["parse_errors"].append(f"{pmid}: {str(e)[:100]}")
                print(f"  [{doc_idx}] Integrity error for {pmid}: {str(e)[:100]}")
            except Exception as e:
                session.rollback()
                summary["parse_errors"].append(f"{pmid}: {type(e).__name__}: {str(e)[:100]}")
                print(f"  [{doc_idx}] Error parsing {pmid}: {type(e).__name__}: {str(e)[:100]}")

        # Final commit
        session.commit()
        print(f"Committed final batch")

    finally:
        session.close()

    summary["elapsed_seconds"] = time.time() - start_time
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print a summary table."""
    print()
    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"MeSH terms queried:         {summary['mesh_terms_queried']}")
    print(f"Unique PMIDs (post-dedup):  {summary['unique_pmids_after_dedup']}")
    print(f"Documents inserted:         {summary['documents_inserted']}")
    print(f"Documents updated:          {summary['documents_updated']}")
    print(f"Chunks inserted:            {summary['chunks_inserted']}")
    print(f"Documents skipped (no abs): {summary['documents_skipped_no_abstract']}")
    print(f"Parse errors:               {len(summary['parse_errors'])}")
    if summary["parse_errors"]:
        for err in summary["parse_errors"][:5]:
            print(f"  - {err}")
        if len(summary["parse_errors"]) > 5:
            print(f"  ... and {len(summary['parse_errors']) - 5} more")
    print(f"Elapsed time:               {summary['elapsed_seconds']:.1f} seconds")
    print("=" * 70)


def verify_ingestion() -> None:
    """Query the database to verify ingestion."""
    print()
    print("Verification:")
    print("-" * 70)
    session = get_session()
    try:
        doc_count = session.query(Document).count()
        chunk_count = session.query(Chunk).count()
        min_year = session.query(Document.year).filter(Document.year.isnot(None)).order_by(Document.year).first()
        max_year = (
            session.query(Document.year)
            .filter(Document.year.isnot(None))
            .order_by(Document.year.desc())
            .first()
        )

        print(f"Total documents in DB:    {doc_count}")
        print(f"Total chunks in DB:       {chunk_count}")
        if min_year:
            print(f"Year range:               {min_year[0]} - {max_year[0] if max_year else 'N/A'}")

        # Year distribution
        year_dist = session.query(
            Document.year, func.count(Document.id)
        ).group_by(Document.year).order_by(Document.year.desc()).limit(10).all()

        if year_dist:
            print()
            print("Top 10 years by document count:")
            for year, count in year_dist:
                if year:
                    print(f"  {year}: {count} documents")

    finally:
        session.close()


def main() -> int:
    try:
        summary = ingest_pubmed()
        print_summary(summary)
        verify_ingestion()
        return 0
    except Exception as e:
        print(f"FATAL ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Need to import func for aggregate
    from sqlalchemy import func

    sys.exit(main())
