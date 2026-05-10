"""Evaluation pipeline: run RAG on test set, compute metrics, generate report.

Loads the clinical Q&A test set, runs each question through the RAG pipeline,
then evaluates with RAGAS metrics and hallucination detection.

Usage:
    cd clinical-rag-assistant
    .venv/bin/python -m evaluate.pipeline                   # full eval
    .venv/bin/python -m evaluate.pipeline --domain trauma   # single domain
    .venv/bin/python -m evaluate.pipeline --dry-run         # test set stats only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add src/ to path when run as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate.clinical_qa_set import ClinicalQASet
from evaluate.hallucination_detector import HallucinationDetector
from evaluate.ragas_metrics import RAGASEvaluator

logger = logging.getLogger(__name__)


def _build_pipeline(model: str = "ollama/llama3.1:8b", max_tokens: int = 500):
    """Build live RAG pipeline from environment. Returns None if deps unavailable."""
    try:
        from dotenv import load_dotenv
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from embed.models import EmbeddingModel
        from generate.litellm_client import LLMClient
        from generate.rag_pipeline import RAGPipeline
        from generate.safety_guardrails import SafetyGuardrails
        from retrieve.hybrid_retriever import BM25Index, HybridRetriever
        from retrieve.query_processor import QueryProcessor
        from retrieve.vector_store import VectorStore
    except ImportError as e:
        logger.warning(f"Cannot build live pipeline: {e}")
        return None

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

    db_url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}"
        f"/{os.environ['POSTGRES_DB']}"
    )
    engine = create_engine(db_url)
    session = sessionmaker(bind=engine)()

    embedding_model = EmbeddingModel(
        model_name="pubmedbert-base-uncased-abstract", device="cpu"
    )
    embedding_model.embed_query("test")  # force load

    llm = LLMClient(model=model, temperature=0.1, max_tokens=max_tokens)
    logger.info(f"LLM: {model}")

    vector_store = VectorStore(session)
    bm25_index = BM25Index.from_db(session)

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

    pipeline = RAGPipeline(
        llm=llm,
        retriever=retriever,
        guardrails=guardrails,
        embedding_model=embedding_model,
        query_processor=query_processor,
    )
    return pipeline, session


def run_evaluation(
    qa_set: ClinicalQASet,
    pipeline=None,
    domain: str | None = None,
    output_dir: str = "metrics",
    limit: int | None = None,
) -> dict[str, Any]:
    """Run full evaluation: RAG answers + RAGAS metrics + hallucination detection.

    Args:
        qa_set: Loaded Q&A test set.
        pipeline: Live RAG pipeline (if None, skips generation and uses dry-run).
        domain: Optional domain filter.
        output_dir: Directory for output reports.

    Returns:
        Evaluation report dict.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Filter Q&A pairs
    if domain:
        pairs = qa_set.filter_by_domain(domain)
        logger.info(f"Filtered to domain '{domain}': {len(pairs)} pairs")
    else:
        pairs = qa_set.get_all()
        logger.info(f"Full test set: {len(pairs)} pairs")

    # Apply limit: sample evenly across domains
    if limit and limit < len(pairs):
        from collections import defaultdict
        by_domain: dict[str, list] = defaultdict(list)
        for p in pairs:
            by_domain[p.get("domain", "unknown")].append(p)
        sampled: list = []
        per_domain = max(1, limit // len(by_domain))
        for d in sorted(by_domain):
            sampled.extend(by_domain[d][:per_domain])
        # Fill remaining slots from underrepresented domains
        while len(sampled) < limit:
            for d in sorted(by_domain):
                if len([s for s in sampled if s.get("domain") == d]) < len(by_domain[d]):
                    remaining = [p for p in by_domain[d] if p not in sampled]
                    if remaining:
                        sampled.append(remaining[0])
                        if len(sampled) >= limit:
                            break
            else:
                break
        pairs = sampled[:limit]
        logger.info(f"Sampled {len(pairs)} pairs across {len(by_domain)} domains (limit={limit})")

    if not pairs:
        logger.warning("No Q&A pairs found. Exiting.")
        return {"error": "no_pairs"}

    # Initialize evaluators
    ragas = RAGASEvaluator()
    hallucination = HallucinationDetector(threshold=0.35)

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    per_question_results: list[dict[str, Any]] = []

    t_start = time.perf_counter()

    for i, pair in enumerate(pairs):
        q = pair["question"]
        expected = pair.get("expected_answer_summary", "")
        qa_id = pair.get("id", f"q{i}")
        domain_tag = pair.get("domain", "unknown")
        difficulty = pair.get("difficulty", "medium")

        logger.info(f"[{i+1}/{len(pairs)}] {qa_id} ({domain_tag}/{difficulty}): {q[:80]}...")

        if pipeline is None:
            # Dry-run: no generation
            per_question_results.append({
                "id": qa_id,
                "question": q,
                "domain": domain_tag,
                "difficulty": difficulty,
                "status": "dry_run",
            })
            continue

        # Run RAG pipeline
        t0 = time.perf_counter()
        try:
            result = pipeline.answer(q, top_k=5)
        except Exception as e:
            logger.error(f"ERROR: {e}")
            per_question_results.append({
                "id": qa_id,
                "question": q,
                "domain": domain_tag,
                "difficulty": difficulty,
                "status": "error",
                "error": str(e),
            })
            continue

        latency = time.perf_counter() - t0
        answer_text = result.get("answer", "")
        source_texts = [s.get("text_preview", s.get("text", "")) for s in result.get("sources", [])]

        questions.append(q)
        answers.append(answer_text)
        contexts.append(source_texts)
        ground_truths.append(expected)

        # Hallucination check
        h_result = hallucination.detect_hallucinations(answer_text, source_texts)
        hallucination.log_evaluation(
            answer_text,
            is_hallucinating=h_result["is_hallucinating"],
            details=h_result,
        )

        per_question_results.append({
            "id": qa_id,
            "question": q,
            "domain": domain_tag,
            "difficulty": difficulty,
            "status": "ok",
            "latency_s": round(latency, 2),
            "confidence": result.get("confidence"),
            "grounding_score": result.get("grounding_score"),
            "is_safe": result.get("is_safe"),
            "num_sources": len(result.get("sources", [])),
            "is_hallucinating": h_result["is_hallucinating"],
            "hallucination_rate": h_result["hallucination_rate"],
            "answer_preview": answer_text[:200],
        })

        logger.info(
            f"conf={result.get('confidence', 0):.3f} "
            f"ground={result.get('grounding_score', 0):.3f} "
            f"halluc={h_result['is_hallucinating']} "
            f"({latency:.1f}s)"
        )

    total_time = time.perf_counter() - t_start

    # Compute RAGAS metrics (only if we have answers)
    ragas_scores: dict[str, Any] = {}
    if questions and answers and contexts:
        ragas_scores = ragas.evaluate(questions, answers, contexts, ground_truths)
        logger.info(f"RAGAS scores:")
        logger.info(f"faithfulness: {ragas_scores['faithfulness']:.4f}")
        logger.info(f"answer_relevance: {ragas_scores['answer_relevance']:.4f}")
        logger.info(f"context_precision: {ragas_scores['context_precision']:.4f}")

    # Aggregate stats
    ok_results = [r for r in per_question_results if r.get("status") == "ok"]
    halluc_rate = hallucination.get_hallucination_rate()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": round(total_time, 2),
        "num_questions": len(pairs),
        "num_answered": len(ok_results),
        "num_errors": sum(1 for r in per_question_results if r.get("status") == "error"),
        "domain_filter": domain,
        "ragas": ragas_scores,
        "hallucination_rate": round(halluc_rate, 4),
        "meets_hallucination_target": halluc_rate <= 0.02,
        "avg_confidence": round(
            sum(r.get("confidence", 0) or 0 for r in ok_results)
            / max(len(ok_results), 1),
            4,
        ),
        "avg_grounding": round(
            sum(r.get("grounding_score", 0) or 0 for r in ok_results)
            / max(len(ok_results), 1),
            4,
        ),
        "avg_latency_s": round(
            sum(r.get("latency_s", 0) for r in ok_results)
            / max(len(ok_results), 1),
            2,
        ),
        "per_question": per_question_results,
    }

    # Save reports
    report_path = out_path / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    hallucination.save_report(out_path / "hallucination_report.json")
    logger.info(f"Hallucination report: {out_path / 'hallucination_report.json'}")

    # Summary
    logger.info(f"=== EVALUATION SUMMARY ===")
    logger.info(f"Questions: {len(pairs)} | Answered: {len(ok_results)}")
    logger.info(f"Hallucination rate: {halluc_rate:.2%} (target: <2%)")
    if ragas_scores:
        logger.info(f"Faithfulness: {ragas_scores['faithfulness']:.4f}")
        logger.info(f"Answer relevance: {ragas_scores['answer_relevance']:.4f}")
        logger.info(f"Context precision: {ragas_scores['context_precision']:.4f}")
    logger.info(f"Total time: {total_time:.1f}s")

    return report


def main():
    parser = argparse.ArgumentParser(description="Clinical RAG evaluation pipeline")
    parser.add_argument(
        "--domain", type=str, default=None, help="Filter Q&A pairs by domain"
    )
    parser.add_argument(
        "--qa-path",
        type=str,
        default="data/qa_test_set.json",
        help="Path to Q&A test set JSON",
    )
    parser.add_argument(
        "--output-dir", type=str, default="metrics", help="Output directory for reports"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/llama3.1:8b",
        help="LLM model string (e.g., ollama/llama3.1:8b, openai/gpt-4o)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Max tokens for LLM generation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N questions (samples evenly across domains)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load test set and show stats without running RAG",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CLINICAL RAG EVALUATION PIPELINE")
    logger.info("=" * 60)

    # Load test set
    qa_set = ClinicalQASet(args.qa_path)
    logger.info(f"Loaded: {qa_set}")
    logger.info(f"Domains: {qa_set.get_domains()}")

    if args.dry_run:
        logger.info("[DRY RUN] Showing test set stats only.")
        for d in qa_set.get_domains():
            pairs = qa_set.filter_by_domain(d)
            diffs = {p.get("difficulty", "?") for p in pairs}
            logger.info(f"{d}: {len(pairs)} pairs ({', '.join(sorted(diffs))})")
        run_evaluation(qa_set, pipeline=None, domain=args.domain, output_dir=args.output_dir)
        return

    # Build live pipeline
    logger.info(f"Building live pipeline (model={args.model})...")
    result = _build_pipeline(model=args.model, max_tokens=args.max_tokens)
    if result is None:
        logger.error("Failed to build pipeline. Run with --dry-run for stats only.")
        sys.exit(1)

    pipeline, session = result

    # Use model name in output dir to avoid overwriting across runs
    model_slug = args.model.replace("/", "_").replace(":", "_")
    out_dir = os.path.join(args.output_dir, model_slug)

    try:
        run_evaluation(
            qa_set,
            pipeline=pipeline,
            domain=args.domain,
            output_dir=out_dir,
            limit=args.limit,
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
