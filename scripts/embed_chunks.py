#!/usr/bin/env python3
"""Phase 3: Embed all chunks in the database with PubMedBERT.

Reads chunks from PostgreSQL, encodes with sentence-transformers,
and writes embedding vectors back to chunks.embedding.
Idempotent: re-running only embeds chunks with NULL embeddings.

Usage:
    source .venv/bin/activate
    python scripts/embed_chunks.py              # embed missing only
    python scripts/embed_chunks.py --all        # re-embed everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load env and add src to path
repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")
sys.path.insert(0, str(repo_root / "src"))

from config_loader import load_config
from db.connection import get_session
from embed.models import EmbeddingModel
from embed.embeddings import batch_embed_db


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed chunks with PubMedBERT")
    parser.add_argument("--all", action="store_true", help="Re-embed all chunks (not just missing)")
    args = parser.parse_args()

    cfg = load_config()

    print(f"Embedding model: {cfg.model.embedding.model}")
    print(f"Device: {cfg.model.embedding.device}")
    print(f"Batch size: {cfg.model.embedding.batch_size}")
    print(f"Mode: {'all chunks' if args.all else 'missing embeddings only'}")
    print()

    # Initialize model
    print("Loading embedding model...")
    model = EmbeddingModel(
        model_name=cfg.model.embedding.model,
        device=cfg.model.embedding.device,
    )

    # Embed
    session = get_session()
    try:
        print("Embedding chunks...")
        summary = batch_embed_db(
            session=session,
            embedding_model=model,
            batch_size=cfg.model.embedding.batch_size,
            only_missing=not args.all,
        )

        print()
        print("=" * 60)
        print("EMBEDDING SUMMARY")
        print("=" * 60)
        print(f"Total chunks queried:  {summary['total_chunks']}")
        print(f"Chunks embedded:       {summary['embedded']}")
        print(f"Chunks skipped:        {summary['skipped']}")
        print(f"Elapsed time:          {summary['elapsed_seconds']:.1f}s")
        if summary["embedded"] > 0:
            rate = summary["embedded"] / summary["elapsed_seconds"]
            print(f"Rate:                  {rate:.1f} chunks/sec")
        print("=" * 60)

        # Verify
        from sqlalchemy import text
        total = session.execute(text("SELECT COUNT(*) FROM chunks")).scalar()
        embedded = session.execute(
            text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        ).scalar()
        print(f"\nDB status: {embedded}/{total} chunks have embeddings")

        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
