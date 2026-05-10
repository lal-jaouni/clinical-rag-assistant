"""Baseline LLM evaluation: retrieval + safety + formatting.

Instead of calling an external LLM via API, this script:
1. Retrieves chunks from the live DB (real retrieval)
2. Builds the full prompt (real prompt templates)
3. Writes the prompt to a file for the LLM to answer inline
4. Reads pre-generated answers from a JSON file
5. Runs safety guardrails + output formatting (real safety pipeline)

This isolates retrieval quality and safety system behavior with a
known-good LLM baseline.

Usage:
    # Step 1: Generate prompts
    .venv/bin/python scripts/baseline_e2e.py --generate-prompts

    # Step 2: (Baseline answers are filled in by the agent)

    # Step 3: Evaluate answers
    .venv/bin/python scripts/baseline_e2e.py --evaluate
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from embed.models import EmbeddingModel
from generate.output_formatter import format_response
from generate.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from generate.safety_guardrails import SafetyGuardrails
from retrieve.vector_store import VectorStore
from retrieve.hybrid_retriever import BM25Index, HybridRetriever
from retrieve.query_processor import QueryProcessor

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DB_URL = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}"
    f"/{os.environ['POSTGRES_DB']}"
)

TEST_QUERIES = [
    "What triggers activation of a massive transfusion protocol?",
    "What is the role of tranexamic acid in trauma?",
    "What are the FDA requirements for AI-based medical devices?",
    "What is the recommended ratio of plasma to red blood cells in massive transfusion?",
    "Tell me about quantum computing applications in cooking",
]

BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "baseline_eval")


def setup_retrieval():
    """Initialize DB, embedding model, and hybrid retriever."""
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    embedding_model = EmbeddingModel(model_name="pubmedbert-base-uncased-abstract", device="cpu")
    # Force load
    embedding_model.embed_query("test")

    vector_store = VectorStore(session)
    bm25_index = BM25Index.from_db(session)
    retriever = HybridRetriever(vector_store=vector_store, bm25_index=bm25_index)
    query_processor = QueryProcessor()

    return session, embedding_model, retriever, query_processor


def generate_prompts():
    """Retrieve chunks and build prompts for each query."""
    os.makedirs(BASELINE_DIR, exist_ok=True)

    print("Setting up retrieval pipeline...")
    session, embedding_model, retriever, query_processor = setup_retrieval()

    prompts_data = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n--- Query {i}: {query}")

        # Query expansion
        qp_result = query_processor.process(query)
        expanded = qp_result["expanded"]
        print(f"  Expanded: {expanded[:100]}...")

        # Retrieve
        query_embedding = embedding_model.embed_query(expanded)
        chunks = retriever.retrieve(
            query=expanded,
            query_embedding=query_embedding,
            top_k=5,
        )

        print(f"  Retrieved {len(chunks)} chunks:")
        for j, c in enumerate(chunks):
            print(f"    [{j+1}] {c.get('source_type','?')} {c.get('source_id','?')} -- {c.get('title','')[:60]}")
            print(f"        Score: {c.get('rrf_score', c.get('score', '?'))}")
            print(f"        Text: {c.get('text','')[:100]}...")

        # Build prompt
        user_prompt = build_user_prompt(query, chunks)

        prompts_data.append({
            "query_index": i,
            "query": query,
            "expanded_query": expanded,
            "chunks": chunks,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "baseline_answer": "",  # To be filled in
        })

    # Save prompts
    prompts_path = os.path.join(BASELINE_DIR, "prompts.json")
    with open(prompts_path, "w") as f:
        json.dump(prompts_data, f, indent=2, default=str)
    print(f"\nPrompts saved to {prompts_path}")

    # Also save human-readable version
    readable_path = os.path.join(BASELINE_DIR, "prompts_readable.txt")
    with open(readable_path, "w") as f:
        for p in prompts_data:
            f.write(f"{'='*70}\n")
            f.write(f"QUERY {p['query_index']}: {p['query']}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"SYSTEM PROMPT:\n{p['system_prompt']}\n\n")
            f.write(f"USER PROMPT:\n{p['user_prompt']}\n\n")
    print(f"Readable prompts saved to {readable_path}")

    session.close()
    return prompts_data


def evaluate_answers():
    """Run safety guardrails on baseline LLM answers."""
    answers_path = os.path.join(BASELINE_DIR, "baseline_answers.json")
    if not os.path.exists(answers_path):
        print(f"ERROR: {answers_path} not found. Generate it first.")
        sys.exit(1)

    with open(answers_path) as f:
        answers_data = json.load(f)

    embedding_model = EmbeddingModel(model_name="pubmedbert-base-uncased-abstract", device="cpu")
    embedding_model.embed_query("test")  # force load

    guardrails = SafetyGuardrails(
        confidence_threshold=0.7,
        grounding_threshold=0.50,
        embedding_model=embedding_model,
    )

    results = []
    print("="*70)
    print("  BASELINE LLM EVALUATION")
    print("="*70)

    for entry in answers_data:
        query = entry["query"]
        answer = entry["baseline_answer"]
        chunks = entry["chunks"]

        print(f"\n--- Query {entry['query_index']}: {query}")

        # Run safety validation
        safety = guardrails.validate(response=answer, source_chunks=chunks)

        # Format response
        result = format_response(
            answer=safety.final_response,
            source_chunks=chunks,
            confidence=safety.confidence,
            grounding_score=safety.grounding_score,
            latency_ms=0,
            model="baseline-llm",
            is_safe=safety.is_safe,
            override_reason=safety.override_reason,
        )

        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Grounding: {result['grounding_score']:.3f}")
        print(f"  Safe: {result['is_safe']}")
        print(f"  Citations: {result['cited_indices']}")
        if result['override_reason']:
            print(f"  Override: {result['override_reason']}")
        print(f"  Answer: {result['answer'][:300]}...")

        results.append({
            "query": query,
            "confidence": result["confidence"],
            "grounding_score": result["grounding_score"],
            "is_safe": result["is_safe"],
            "cited_indices": result["cited_indices"],
            "override_reason": result["override_reason"],
            "answer_preview": result["answer"][:300],
            "num_sources": len(result["sources"]),
            "has_citations": safety.has_citations,
            "has_direct_advice": safety.has_direct_advice,
            "has_uncertainty": safety.has_uncertainty,
        })

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    avg_ground = sum(r["grounding_score"] for r in results) / len(results)
    safe_count = sum(1 for r in results if r["is_safe"])
    cited_count = sum(1 for r in results if r["has_citations"])

    print(f"  Queries: {len(results)}")
    print(f"  Avg confidence: {avg_conf:.3f}")
    print(f"  Avg grounding: {avg_ground:.3f}")
    print(f"  Safe responses: {safe_count}/{len(results)}")
    print(f"  Responses with citations: {cited_count}/{len(results)}")

    # Save
    eval_path = os.path.join(BASELINE_DIR, "baseline_results.json")
    with open(eval_path, "w") as f:
        json.dump({
            "model": "baseline-llm",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "avg_confidence": round(avg_conf, 3),
            "avg_grounding": round(avg_ground, 3),
            "safe_count": safe_count,
            "cited_count": cited_count,
            "total": len(results),
            "queries": results,
        }, f, indent=2)
    print(f"\n  Results saved to {eval_path}")


if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        evaluate_answers()
    else:
        generate_prompts()
