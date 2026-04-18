"""PubMedQA benchmark evaluation for Clinical RAG Assistant.

Evaluates the generation and safety layers against the PubMedQA expert-labeled
dataset (1,000 biomedical yes/no/maybe questions with gold-standard contexts
and long-form answers).

Two evaluation modes:
1. **Generation-only** (default): Uses PubMedQA's own abstracts as context,
   bypassing retrieval. Tests answer quality, safety guardrails, and
   hallucination detection against a published benchmark.
2. **Full pipeline**: Retrieves from the project's ingested corpus, then
   generates. Tests end-to-end RAG including retrieval relevance.

Metrics reported:
- Answer accuracy (yes/no/maybe classification vs expert label)
- RAGAS scores (faithfulness, answer_relevance, context_precision)
- Hallucination rate (target <2%)
- Safety override rate
- Latency statistics

Usage:
    python -m evaluate.pubmedqa_benchmark --limit 50          # quick test
    python -m evaluate.pubmedqa_benchmark --limit 200         # moderate
    python -m evaluate.pubmedqa_benchmark                     # full 1000
    python -m evaluate.pubmedqa_benchmark --full-pipeline     # end-to-end RAG
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate.hallucination_detector import HallucinationDetector
from evaluate.ragas_metrics import RAGASEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _classify_answer(answer: str) -> str:
    """Extract yes/no/maybe classification from generated answer.

    Looks for explicit yes/no/maybe at the start of the answer or in
    common answer patterns. Falls back to 'maybe' if ambiguous.
    """
    clean = answer.lower().strip()

    # Check first sentence for explicit classification
    first_line = clean.split("\n")[0].split(".")[0].strip()

    # Direct match patterns
    if re.match(r"^(yes|the answer is yes)", first_line):
        return "yes"
    if re.match(r"^(no|the answer is no)", first_line):
        return "no"
    if re.match(r"^(maybe|the answer is maybe|it is unclear|the evidence is mixed)", first_line):
        return "maybe"

    # Search broader answer
    yes_signals = ["yes,", "the answer is yes", "evidence supports", "studies confirm",
                   "findings indicate that", "data suggest that"]
    no_signals = ["no,", "the answer is no", "evidence does not support",
                  "no significant", "did not show", "no evidence"]
    maybe_signals = ["maybe", "unclear", "mixed evidence", "inconclusive",
                     "further research", "insufficient evidence", "cannot be determined"]

    yes_count = sum(1 for s in yes_signals if s in clean)
    no_count = sum(1 for s in no_signals if s in clean)
    maybe_count = sum(1 for s in maybe_signals if s in clean)

    counts = {"yes": yes_count, "no": no_count, "maybe": maybe_count}
    best = max(counts, key=counts.get)  # type: ignore[arg-type]
    if counts[best] == 0:
        return "maybe"  # default when no signal found
    return best


def _build_llm_client(model: str, max_tokens: int):
    """Build LLM client for generation."""
    from generate.litellm_client import LLMClient
    return LLMClient(model=model, temperature=0.1, max_tokens=max_tokens)


def _build_safety(embedding_model=None):
    """Build SafetyGuardrails."""
    from generate.safety_guardrails import SafetyGuardrails
    return SafetyGuardrails(
        confidence_threshold=0.7,
        grounding_threshold=0.50,
        embedding_model=embedding_model,
    )


def _build_full_pipeline(model: str, max_tokens: int):
    """Build full RAG pipeline for end-to-end evaluation."""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from embed.models import EmbeddingModel
    from generate.rag_pipeline import RAGPipeline
    from retrieve.hybrid_retriever import BM25Index, HybridRetriever
    from retrieve.query_processor import QueryProcessor
    from retrieve.vector_store import VectorStore

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
    embedding_model.embed_query("test")

    llm = _build_llm_client(model, max_tokens)
    vector_store = VectorStore(session)
    bm25_index = BM25Index.from_db(session)
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        vector_weight=0.6,
        bm25_weight=0.4,
    )
    query_processor = QueryProcessor()
    guardrails = _build_safety(embedding_model)

    pipeline = RAGPipeline(
        llm=llm,
        retriever=retriever,
        guardrails=guardrails,
        embedding_model=embedding_model,
        query_processor=query_processor,
    )
    return pipeline, session


def _generation_only_answer(
    question: str,
    contexts: list[str],
    llm_client,
    guardrails,
) -> dict[str, Any]:
    """Generate answer using PubMedQA's own context (bypass retrieval)."""
    from generate.prompt_templates import build_user_prompt, SYSTEM_PROMPT

    # Build chunks as dicts matching what build_user_prompt and SafetyGuardrails expect
    chunks = []
    for i, ctx in enumerate(contexts):
        chunks.append({
            "text": ctx,
            "source_type": "pubmed",
            "source_id": f"context_{i}",
            "title": f"PubMedQA Context {i+1}",
        })

    user_prompt = build_user_prompt(question, chunks)

    t0 = time.perf_counter()
    llm_response = llm_client.complete(SYSTEM_PROMPT, user_prompt)
    latency = time.perf_counter() - t0

    answer_text = llm_response.text if hasattr(llm_response, "text") else str(llm_response)

    # Run safety guardrails (expects list[dict] with "text" key, returns SafetyResult dataclass)
    safety_result = guardrails.validate(answer_text, chunks)

    return {
        "answer": safety_result.final_response,
        "raw_answer": answer_text,
        "confidence": safety_result.confidence,
        "grounding_score": safety_result.grounding_score,
        "is_safe": safety_result.is_safe,
        "override_reason": safety_result.override_reason,
        "latency_s": round(latency, 2),
        "sources": contexts,
    }


def run_pubmedqa_benchmark(
    qa_path: str = "data/pubmedqa_test_set.json",
    model: str = "ollama/llama3.1:8b",
    max_tokens: int = 400,
    limit: int | None = None,
    full_pipeline: bool = False,
    output_dir: str = "metrics/pubmedqa",
) -> dict[str, Any]:
    """Run PubMedQA benchmark evaluation.

    Args:
        qa_path: Path to converted PubMedQA test set JSON.
        model: LLM model string.
        max_tokens: Max generation tokens.
        limit: Limit number of questions (None = all 1000).
        full_pipeline: If True, use full RAG pipeline (retrieval + generation).
        output_dir: Directory for output reports.

    Returns:
        Benchmark report dict.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load PubMedQA test set
    with open(qa_path) as f:
        data = json.load(f)

    pairs = data["qa_pairs"]
    if limit:
        pairs = pairs[:limit]

    logger.info(f"PubMedQA benchmark: {len(pairs)} questions (model={model})")

    # Build components
    pipeline = None
    session = None
    llm_client = None
    guardrails = None

    if full_pipeline:
        logger.info("Building full RAG pipeline...")
        pipeline, session = _build_full_pipeline(model, max_tokens)
    else:
        logger.info("Generation-only mode (using PubMedQA contexts)")
        llm_client = _build_llm_client(model, max_tokens)
        guardrails = _build_safety()

    ragas = RAGASEvaluator()
    hallucination = HallucinationDetector(threshold=0.35)

    # Tracking
    results: list[dict[str, Any]] = []
    correct = 0
    total_answered = 0
    label_counts = {"yes": 0, "no": 0, "maybe": 0}
    pred_counts = {"yes": 0, "no": 0, "maybe": 0}
    confusion: dict[str, dict[str, int]] = {
        "yes": {"yes": 0, "no": 0, "maybe": 0},
        "no": {"yes": 0, "no": 0, "maybe": 0},
        "maybe": {"yes": 0, "no": 0, "maybe": 0},
    }

    questions_list: list[str] = []
    answers_list: list[str] = []
    contexts_list: list[list[str]] = []
    ground_truths_list: list[str] = []

    t_start = time.perf_counter()

    for i, pair in enumerate(pairs):
        q = pair["question"]
        expected_label = pair.get("pubmedqa_label", "maybe")
        expected_answer = pair.get("expected_answer_summary", "")
        contexts = pair.get("pubmedqa_contexts", [])
        qa_id = pair["id"]

        logger.info(f"[{i+1}/{len(pairs)}] {qa_id}: {q[:70]}...")

        try:
            if full_pipeline and pipeline:
                result = pipeline.answer(q, top_k=5)
                answer_text = result.get("answer", "")
                source_texts = [s.get("text_preview", s.get("text", "")) for s in result.get("sources", [])]
                latency = result.get("latency_s", 0)
                confidence = result.get("confidence", 0)
                grounding = result.get("grounding_score", 0)
                is_safe = result.get("is_safe", True)
                override_reason = result.get("override_reason")
            else:
                result = _generation_only_answer(q, contexts, llm_client, guardrails)
                answer_text = result["answer"]
                source_texts = contexts
                latency = result["latency_s"]
                confidence = result["confidence"]
                grounding = result["grounding_score"]
                is_safe = result["is_safe"]
                override_reason = result["override_reason"]

        except Exception as e:
            logger.error(f"ERROR on {qa_id}: {e}")
            results.append({
                "id": qa_id,
                "question": q,
                "status": "error",
                "error": str(e),
                "expected_label": expected_label,
            })
            continue

        # Classify answer
        predicted_label = _classify_answer(answer_text)
        is_correct = predicted_label == expected_label

        if is_correct:
            correct += 1
        total_answered += 1

        label_counts[expected_label] = label_counts.get(expected_label, 0) + 1
        pred_counts[predicted_label] = pred_counts.get(predicted_label, 0) + 1
        confusion[expected_label][predicted_label] += 1

        # Track for RAGAS
        questions_list.append(q)
        answers_list.append(answer_text)
        contexts_list.append(source_texts)
        ground_truths_list.append(expected_answer)

        # Hallucination check
        h_result = hallucination.detect_hallucinations(answer_text, source_texts)
        hallucination.log_evaluation(
            answer_text,
            is_hallucinating=h_result["is_hallucinating"],
            details=h_result,
        )

        results.append({
            "id": qa_id,
            "question": q,
            "status": "ok",
            "expected_label": expected_label,
            "predicted_label": predicted_label,
            "is_correct": is_correct,
            "confidence": confidence,
            "grounding_score": grounding,
            "is_safe": is_safe,
            "override_reason": override_reason,
            "is_hallucinating": h_result["is_hallucinating"],
            "latency_s": latency,
            "answer_preview": answer_text[:300],
        })

        logger.info(
            f"  expected={expected_label} predicted={predicted_label} "
            f"{'CORRECT' if is_correct else 'WRONG'} "
            f"conf={confidence:.3f} ground={grounding:.3f} "
            f"halluc={h_result['is_hallucinating']} ({latency:.1f}s)"
        )

    total_time = time.perf_counter() - t_start

    # RAGAS metrics
    ragas_scores: dict[str, Any] = {}
    if questions_list:
        ragas_scores = ragas.evaluate(questions_list, answers_list, contexts_list, ground_truths_list)

    # Aggregate
    accuracy = correct / max(total_answered, 1)
    halluc_rate = hallucination.get_hallucination_rate()
    ok_results = [r for r in results if r.get("status") == "ok"]
    safety_overrides = sum(1 for r in ok_results if not r.get("is_safe", True))

    report = {
        "benchmark": "PubMedQA (PQA-L, expert-labeled)",
        "mode": "full_pipeline" if full_pipeline else "generation_only",
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": round(total_time, 2),
        "num_questions": len(pairs),
        "num_answered": total_answered,
        "num_errors": len(pairs) - total_answered,
        "accuracy": {
            "overall": round(accuracy, 4),
            "correct": correct,
            "total": total_answered,
            "by_label": {
                label: {
                    "total": label_counts.get(label, 0),
                    "correct": confusion.get(label, {}).get(label, 0),
                    "accuracy": round(
                        confusion.get(label, {}).get(label, 0)
                        / max(label_counts.get(label, 0), 1),
                        4,
                    ),
                }
                for label in ["yes", "no", "maybe"]
            },
            "confusion_matrix": confusion,
        },
        "ragas": ragas_scores,
        "hallucination_rate": round(halluc_rate, 4),
        "meets_hallucination_target": halluc_rate <= 0.02,
        "safety_override_rate": round(safety_overrides / max(total_answered, 1), 4),
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
        "per_question": results,
    }

    # Save
    report_path = out_path / "pubmedqa_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    hallucination.save_report(out_path / "pubmedqa_hallucination_report.json")

    # Summary
    logger.info("=" * 60)
    logger.info("PUBMEDQA BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Mode: {'Full Pipeline' if full_pipeline else 'Generation Only'}")
    logger.info(f"Questions: {len(pairs)} | Answered: {total_answered}")
    logger.info(f"Accuracy: {accuracy:.1%} ({correct}/{total_answered})")
    for label in ["yes", "no", "maybe"]:
        la = report["accuracy"]["by_label"][label]
        logger.info(f"  {label}: {la['accuracy']:.1%} ({la['correct']}/{la['total']})")
    logger.info(f"Hallucination rate: {halluc_rate:.2%} (target: <2%)")
    logger.info(f"Safety overrides: {safety_overrides}/{total_answered}")
    if ragas_scores:
        logger.info(f"Faithfulness: {ragas_scores.get('faithfulness', 0):.4f}")
        logger.info(f"Answer relevance: {ragas_scores.get('answer_relevance', 0):.4f}")
        logger.info(f"Context precision: {ragas_scores.get('context_precision', 0):.4f}")
    logger.info(f"Avg latency: {report['avg_latency_s']}s")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Report: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="PubMedQA benchmark evaluation")
    parser.add_argument(
        "--qa-path",
        default="data/pubmedqa_test_set.json",
        help="Path to converted PubMedQA test set JSON",
    )
    parser.add_argument(
        "--model",
        default="ollama/llama3.1:8b",
        help="LLM model string",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N questions",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Use full RAG pipeline (retrieval + generation) instead of PubMedQA contexts",
    )
    parser.add_argument(
        "--output-dir",
        default="metrics/pubmedqa",
        help="Output directory for reports",
    )
    args = parser.parse_args()

    run_pubmedqa_benchmark(
        qa_path=args.qa_path,
        model=args.model,
        max_tokens=args.max_tokens,
        limit=args.limit,
        full_pipeline=args.full_pipeline,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
