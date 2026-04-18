"""Retrieve chunks for all Q&A test set questions (no LLM call).

Outputs a JSON file with questions + retrieved chunks + built prompts
for offline LLM evaluation.

Usage:
    cd clinical-rag-assistant
    PYTHONPATH=src .venv/bin/python scripts/retrieve_for_eval.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from embed.models import EmbeddingModel
from evaluate.clinical_qa_set import ClinicalQASet
from generate.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from retrieve.hybrid_retriever import BM25Index, HybridRetriever
from retrieve.query_processor import QueryProcessor
from retrieve.vector_store import VectorStore

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DB_URL = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}"
    f"/{os.environ['POSTGRES_DB']}"
)


def main():
    print("Loading models and building retriever...")
    engine = create_engine(DB_URL)
    session = sessionmaker(bind=engine)()

    embedding_model = EmbeddingModel(model_name="pubmedbert-base-uncased-abstract", device="cpu")
    embedding_model.embed_query("test")

    vector_store = VectorStore(session)
    bm25_index = BM25Index.from_db(session)
    retriever = HybridRetriever(
        vector_store=vector_store, bm25_index=bm25_index,
        vector_weight=0.6, bm25_weight=0.4,
    )
    query_processor = QueryProcessor()

    qa_set = ClinicalQASet("data/qa_test_set.json")
    pairs = qa_set.get_all()
    print(f"Loaded {len(pairs)} Q&A pairs")

    results = []
    for i, pair in enumerate(pairs):
        q = pair["question"]
        qa_id = pair.get("id", f"q{i}")
        print(f"  [{i+1}/{len(pairs)}] {qa_id}: {q[:70]}...")

        # Query expansion
        qp_result = query_processor.process(q)
        expanded = qp_result["expanded"]

        # Retrieve
        query_embedding = embedding_model.embed_query(expanded)
        chunks = retriever.retrieve(
            query=expanded, query_embedding=query_embedding, top_k=5
        )

        # Build prompt
        user_prompt = build_user_prompt(q, chunks)

        results.append({
            "id": qa_id,
            "question": q,
            "domain": pair.get("domain", ""),
            "difficulty": pair.get("difficulty", ""),
            "expected_answer_summary": pair.get("expected_answer_summary", ""),
            "source_ids": pair.get("source_ids", []),
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "retrieved_chunks": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "text": c.get("text", ""),
                    "source_type": c.get("source_type", ""),
                    "source_id": c.get("source_id", ""),
                    "title": c.get("title", ""),
                    "year": c.get("year"),
                    "url": c.get("url", ""),
                }
                for c in chunks
            ],
        })

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_retrieval.json")
    with open(out_path, "w") as f:
        json.dump({"pairs": results, "system_prompt": SYSTEM_PROMPT}, f, indent=2)
    print(f"\nSaved {len(results)} retrieval results to {out_path}")

    session.close()


if __name__ == "__main__":
    main()
