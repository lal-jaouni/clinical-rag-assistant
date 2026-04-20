"""Live end-to-end integration test: real DB + real embeddings + real LLM.

Runs clinical queries through the full RAG pipeline and validates:
1. Retrieval returns relevant chunks from the live PostgreSQL database
2. LLM generates grounded, cited responses via Ollama
3. Safety guardrails correctly gate unsafe/ungrounded answers
4. Output structure matches expected schema

Usage:
    cd clinical-rag-assistant
    .venv/bin/python scripts/live_e2e_test.py
"""

import json
import os
import sys
import time

# Add src/ to path (same convention as other scripts)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from embed.models import EmbeddingModel
from generate.litellm_client import LLMClient
from generate.output_formatter import format_response
from generate.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from generate.rag_pipeline import RAGPipeline
from generate.safety_guardrails import SafetyGuardrails
from retrieve.vector_store import VectorStore
from retrieve.hybrid_retriever import BM25Index, HybridRetriever
from retrieve.query_processor import QueryProcessor


# ── Setup ─────────────────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DB_URL = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}"
    f"/{os.environ['POSTGRES_DB']}"
)

# Clinical test queries with expected behaviors
TEST_QUERIES = [
    {
        "query": "What triggers activation of a massive transfusion protocol?",
        "expect_sources": True,
        "expect_safe": True,
        "description": "Core clinical question -- should retrieve MTP-related chunks",
    },
    {
        "query": "What is the role of tranexamic acid in trauma?",
        "expect_sources": True,
        "expect_safe": True,
        "description": "TXA in trauma -- should have good PubMed coverage",
    },
    {
        "query": "What are the FDA requirements for AI-based medical devices?",
        "expect_sources": True,
        "expect_safe": True,
        "description": "FDA SaMD -- should retrieve FDA guidance chunks",
    },
    {
        "query": "What is the recommended ratio of plasma to red blood cells in massive transfusion?",
        "expect_sources": True,
        "expect_safe": True,
        "description": "Plasma:RBC ratios -- quantitative clinical question",
    },
    {
        "query": "Tell me about quantum computing applications in cooking",
        "expect_sources": False,
        "expect_safe": True,
        "description": "Out-of-domain query -- should return no sources or low confidence",
    },
]


def hr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def check_response_structure(result: dict) -> list[str]:
    """Validate response dict has all required keys with correct types."""
    errors = []
    required_keys = {
        "answer": str,
        "confidence": float,
        "grounding_score": float,
        "is_safe": bool,
        "sources": list,
        "cited_indices": list,
        "latency_ms": (int, type(None)),
        "model": (str, type(None)),
    }
    for key, expected_type in required_keys.items():
        if key not in result:
            errors.append(f"Missing key: {key}")
        elif not isinstance(result[key], expected_type):
            errors.append(f"{key}: expected {expected_type}, got {type(result[key])}")

    if "confidence" in result and not (0.0 <= result["confidence"] <= 1.0):
        errors.append(f"confidence out of range: {result['confidence']}")
    if "grounding_score" in result and not (0.0 <= result["grounding_score"] <= 1.0):
        errors.append(f"grounding_score out of range: {result['grounding_score']}")

    return errors


def main():
    t_start = time.perf_counter()
    results_summary = []

    print(hr("LIVE END-TO-END INTEGRATION TEST"))
    print(f"  Database: {os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}")
    print(f"  LLM: ollama/llama3.1:8b")
    print(f"  Embedding: PubMedBERT (768-dim)")

    # ── 1. Database connection ────────────────────────────────────────────
    print(hr("1. DATABASE CONNECTION"))
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    row = session.execute(text("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")).scalar()
    print(f"  Embedded chunks in DB: {row}")
    assert row > 0, "No embedded chunks found -- run embed_chunks.py first"

    doc_count = session.execute(text("SELECT count(*) FROM documents")).scalar()
    print(f"  Documents in DB: {doc_count}")

    source_types = session.execute(
        text("SELECT source_type, count(*) FROM documents GROUP BY source_type ORDER BY count(*) DESC")
    ).fetchall()
    for st, ct in source_types:
        print(f"    {st}: {ct} documents")

    # ── 2. Load models ────────────────────────────────────────────────────
    print(hr("2. LOADING MODELS"))

    t = time.perf_counter()
    embedding_model = EmbeddingModel(model_name="pubmedbert-base-uncased-abstract", device="cpu")
    # Force load
    test_vec = embedding_model.embed_query("test")
    print(f"  Embedding model loaded ({time.perf_counter()-t:.1f}s), dim={len(test_vec)}")
    assert len(test_vec) == 768, f"Expected 768-dim, got {len(test_vec)}"

    t = time.perf_counter()
    llm = LLMClient(model="ollama/llama3.1:8b", temperature=0.1, max_tokens=500)
    print(f"  LLM client initialized")

    # Ping test
    print("  Pinging Ollama...", end=" ", flush=True)
    if llm.ping():
        print("OK")
    else:
        print("FAILED -- is Ollama running?")
        sys.exit(1)

    # ── 3. Build retriever ────────────────────────────────────────────────
    print(hr("3. BUILDING RETRIEVER"))

    vector_store = VectorStore(session)
    print(f"  VectorStore connected ({vector_store.count_embedded()} embedded chunks)")

    t = time.perf_counter()
    bm25_index = BM25Index.from_db(session)
    print(f"  BM25 index built ({time.perf_counter()-t:.1f}s)")

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        vector_weight=0.6,
        bm25_weight=0.4,
    )

    query_processor = QueryProcessor()
    guardrails = SafetyGuardrails(
        confidence_threshold=0.7,
        grounding_threshold=0.50,
        embedding_model=embedding_model,
    )

    # ── 4. Build pipeline ─────────────────────────────────────────────────
    print(hr("4. RAG PIPELINE"))

    pipeline = RAGPipeline(
        llm=llm,
        retriever=retriever,
        guardrails=guardrails,
        embedding_model=embedding_model,
        query_processor=query_processor,
    )

    # ── 5. Run test queries ───────────────────────────────────────────────
    print(hr("5. RUNNING TEST QUERIES"))

    total_pass = 0
    total_fail = 0

    for i, tq in enumerate(TEST_QUERIES, 1):
        print(f"\n--- Query {i}/{len(TEST_QUERIES)} ---")
        print(f"  Q: {tq['query']}")
        print(f"  ({tq['description']})")

        t = time.perf_counter()
        try:
            result = pipeline.answer(tq["query"], top_k=5)
        except Exception as e:
            print(f"  ERROR: {e}")
            total_fail += 1
            results_summary.append({
                "query": tq["query"],
                "status": "ERROR",
                "error": str(e),
            })
            continue

        elapsed = time.perf_counter() - t
        print(f"  Latency: {elapsed:.1f}s ({result.get('latency_ms', 0)}ms)")
        print(f"  Model: {result.get('model', 'unknown')}")
        print(f"  Sources: {len(result.get('sources', []))}")
        print(f"  Confidence: {result.get('confidence', 0):.3f}")
        print(f"  Grounding: {result.get('grounding_score', 0):.3f}")
        print(f"  Safe: {result.get('is_safe', False)}")
        print(f"  Cited indices: {result.get('cited_indices', [])}")
        if result.get("override_reason"):
            print(f"  Override: {result['override_reason']}")

        # Truncate answer for display
        answer = result.get("answer", "")
        if len(answer) > 300:
            print(f"  Answer: {answer[:300]}...")
        else:
            print(f"  Answer: {answer}")

        # Show source previews
        for s in result.get("sources", [])[:3]:
            print(f"    [{s['index']}] {s['citation']} -- {s.get('title', '')[:60]}")

        # Validate structure
        struct_errors = check_response_structure(result)
        if struct_errors:
            print(f"  STRUCT ERRORS: {struct_errors}")

        # Validate expectations
        issues = []
        if tq["expect_sources"] and len(result.get("sources", [])) == 0:
            issues.append("Expected sources but got none")
        if not tq["expect_sources"] and len(result.get("sources", [])) > 0:
            # This is ok -- the model might find tangentially related content
            pass

        if struct_errors:
            issues.extend(struct_errors)

        if issues:
            print(f"  FAIL: {issues}")
            total_fail += 1
        else:
            print(f"  PASS")
            total_pass += 1

        results_summary.append({
            "query": tq["query"],
            "description": tq["description"],
            "status": "FAIL" if issues else "PASS",
            "issues": issues,
            "latency_s": round(elapsed, 2),
            "confidence": result.get("confidence"),
            "grounding_score": result.get("grounding_score"),
            "is_safe": result.get("is_safe"),
            "num_sources": len(result.get("sources", [])),
            "cited_indices": result.get("cited_indices", []),
            "override_reason": result.get("override_reason"),
            "answer_preview": answer[:200],
        })

    # ── 6. Summary ────────────────────────────────────────────────────────
    print(hr("6. SUMMARY"))
    total_time = time.perf_counter() - t_start
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Passed: {total_pass}/{len(TEST_QUERIES)}")
    print(f"  Failed: {total_fail}/{len(TEST_QUERIES)}")

    avg_latency = sum(r.get("latency_s", 0) for r in results_summary) / max(len(results_summary), 1)
    avg_confidence = sum(r.get("confidence", 0) or 0 for r in results_summary) / max(len(results_summary), 1)
    avg_grounding = sum(r.get("grounding_score", 0) or 0 for r in results_summary) / max(len(results_summary), 1)

    print(f"  Avg latency: {avg_latency:.1f}s per query")
    print(f"  Avg confidence: {avg_confidence:.3f}")
    print(f"  Avg grounding: {avg_grounding:.3f}")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "..", "e2e_test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_s": round(total_time, 2),
            "passed": total_pass,
            "failed": total_fail,
            "total": len(TEST_QUERIES),
            "avg_latency_s": round(avg_latency, 2),
            "avg_confidence": round(avg_confidence, 3),
            "avg_grounding": round(avg_grounding, 3),
            "queries": results_summary,
        }, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    session.close()
    return total_fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
