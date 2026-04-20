"""Quick 3-question comparison: Llama 3.1 8B vs Granite 3.1 2B."""

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
from evaluate.hallucination_detector import HallucinationDetector
from evaluate.ragas_metrics import RAGASEvaluator
from generate.litellm_client import LLMClient
from generate.rag_pipeline import RAGPipeline
from generate.safety_guardrails import SafetyGuardrails
from retrieve.hybrid_retriever import BM25Index, HybridRetriever
from retrieve.query_processor import QueryProcessor
from retrieve.vector_store import VectorStore

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DB_URL = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}"
    f"/{os.environ['POSTGRES_DB']}"
)

MODELS = [
    "ollama/llama3.1:8b",
    "ollama/granite3.1-dense:2b",
]


def build_shared_components(session):
    """Build retriever + embedding model (shared across models)."""
    embedding_model = EmbeddingModel(
        model_name="pubmedbert-base-uncased-abstract", device="cpu"
    )
    embedding_model.embed_query("test")

    vector_store = VectorStore(session)
    bm25_index = BM25Index.from_db(session)
    retriever = HybridRetriever(
        vector_store=vector_store, bm25_index=bm25_index,
        vector_weight=0.6, bm25_weight=0.4,
    )
    query_processor = QueryProcessor()
    guardrails = SafetyGuardrails(
        confidence_threshold=0.7, grounding_threshold=0.50,
        embedding_model=embedding_model,
    )
    return embedding_model, retriever, query_processor, guardrails


def run_model(model_name, questions, embedding_model, retriever, query_processor, guardrails):
    """Run a model on the given questions."""
    llm = LLMClient(model=model_name, temperature=0.1, max_tokens=500)
    pipeline = RAGPipeline(
        llm=llm, retriever=retriever, guardrails=guardrails,
        embedding_model=embedding_model, query_processor=query_processor,
    )

    hallucination = HallucinationDetector(threshold=0.7)
    ragas = RAGASEvaluator()
    results = []

    for i, (qa_id, q, expected) in enumerate(questions):
        print(f"  [{model_name}] Q{i+1}: {q[:60]}...")
        t0 = time.perf_counter()
        try:
            result = pipeline.answer(q, top_k=5)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "id": qa_id, "question": q, "model": model_name,
                "status": "error", "error": str(e),
            })
            continue

        latency = time.perf_counter() - t0
        answer = result.get("answer", "")
        sources = [s.get("text_preview", s.get("text", "")) for s in result.get("sources", [])]

        h = hallucination.detect_hallucinations(answer, sources)
        f = ragas.faithfulness(answer, sources)
        r = ragas.answer_relevance(answer, q)

        results.append({
            "id": qa_id, "question": q, "model": model_name,
            "status": "ok",
            "answer_preview": answer[:300],
            "latency_s": round(latency, 1),
            "confidence": result.get("confidence"),
            "grounding": result.get("grounding_score"),
            "is_hallucinating": h["is_hallucinating"],
            "overall_grounding": h["overall_grounding"],
            "faithfulness": round(f, 4),
            "answer_relevance": round(r, 4),
        })

        print(
            f"    {latency:.1f}s | conf={result.get('confidence', 0):.3f} "
            f"ground={h['overall_grounding']:.3f} faith={f:.3f} "
            f"halluc={h['is_hallucinating']}"
        )

    return results


def main():
    print("=" * 60)
    print("  3-QUESTION MODEL COMPARISON")
    print("=" * 60)

    engine = create_engine(DB_URL)
    session = sessionmaker(bind=engine)()

    print("\nLoading shared components...")
    embedding_model, retriever, query_processor, guardrails = build_shared_components(session)

    qa_set = ClinicalQASet(
        os.path.join(os.path.dirname(__file__), "..", "data", "qa_test_set.json")
    )
    all_pairs = qa_set.get_all()

    # First 3 questions
    questions = [
        (p.get("id", f"q{i}"), p["question"], p.get("expected_answer_summary", ""))
        for i, p in enumerate(all_pairs[:3])
    ]

    print(f"\nQuestions:")
    for qa_id, q, _ in questions:
        print(f"  {qa_id}: {q[:80]}")

    all_results = {}
    for model in MODELS:
        print(f"\n{'='*40}")
        print(f"  MODEL: {model}")
        print(f"{'='*40}")
        all_results[model] = run_model(
            model, questions, embedding_model, retriever, query_processor, guardrails
        )

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Llama 3.1 8B':>15} {'Granite 3.1 2B':>15}")
    print("-" * 55)

    for model in MODELS:
        ok = [r for r in all_results[model] if r.get("status") == "ok"]
        if not ok:
            continue
        tag = "Llama 3.1 8B" if "llama" in model else "Granite 3.1 2B"
        avg_lat = sum(r["latency_s"] for r in ok) / len(ok)
        avg_ground = sum(r["overall_grounding"] for r in ok) / len(ok)
        avg_faith = sum(r["faithfulness"] for r in ok) / len(ok)
        avg_rel = sum(r["answer_relevance"] for r in ok) / len(ok)
        halluc_count = sum(1 for r in ok if r["is_hallucinating"])
        errors = sum(1 for r in all_results[model] if r.get("status") == "error")

        if tag == "Llama 3.1 8B":
            llama = (avg_lat, avg_ground, avg_faith, avg_rel, halluc_count, errors, len(ok))
        else:
            granite = (avg_lat, avg_ground, avg_faith, avg_rel, halluc_count, errors, len(ok))

    if 'llama' in dir() or True:
        for model in MODELS:
            ok = [r for r in all_results[model] if r.get("status") == "ok"]
            tag = "Llama 3.1 8B" if "llama" in model else "Granite 3.1 2B"
            if not ok:
                print(f"  {tag}: all errors")
                continue
            avg_lat = sum(r["latency_s"] for r in ok) / len(ok)
            avg_ground = sum(r["overall_grounding"] for r in ok) / len(ok)
            avg_faith = sum(r["faithfulness"] for r in ok) / len(ok)
            avg_rel = sum(r["answer_relevance"] for r in ok) / len(ok)
            halluc_count = sum(1 for r in ok if r["is_hallucinating"])
            errors = sum(1 for r in all_results[model] if r.get("status") == "error")
            print(f"\n  {tag}:")
            print(f"    Answered:        {len(ok)}/3")
            print(f"    Errors:          {errors}")
            print(f"    Avg latency:     {avg_lat:.1f}s")
            print(f"    Avg grounding:   {avg_ground:.3f}")
            print(f"    Avg faithfulness:{avg_faith:.3f}")
            print(f"    Avg relevance:   {avg_rel:.3f}")
            print(f"    Hallucinating:   {halluc_count}/{len(ok)}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", "metrics", "model_comparison_3q.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    session.close()


if __name__ == "__main__":
    main()
