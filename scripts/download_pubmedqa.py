"""Download PubMedQA expert-labeled dataset for benchmark evaluation.

Downloads the PQA-L (labeled) subset: 1,000 expert-annotated yes/no/maybe
questions with PubMed abstracts as context and long-form answers.

Source: https://pubmedqa.github.io/
Paper: Jin et al., "PubMedQA: A Dataset for Biomedical Research Question Answering" (EMNLP 2019)

Usage:
    python scripts/download_pubmedqa.py
    python scripts/download_pubmedqa.py --output data/pubmedqa_labeled.json
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PUBMEDQA_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)


def download_pubmedqa(output_path: str = "data/pubmedqa_labeled.json") -> Path:
    """Download PubMedQA labeled dataset and save as JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        logger.info(f"Already exists: {out}")
        with open(out) as f:
            data = json.load(f)
        logger.info(f"Contains {len(data)} questions")
        return out

    logger.info(f"Downloading PubMedQA labeled set from GitHub...")
    req = urllib.request.Request(PUBMEDQA_URL, headers={"User-Agent": "clinical-rag"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    logger.info(f"Downloaded {len(raw)} questions")

    with open(out, "w") as f:
        json.dump(raw, f, indent=2)

    logger.info(f"Saved to {out}")
    return out


def convert_to_qa_format(
    pubmedqa_path: str = "data/pubmedqa_labeled.json",
    output_path: str = "data/pubmedqa_test_set.json",
) -> Path:
    """Convert PubMedQA to the project's ClinicalQASet format.

    PubMedQA format (per entry):
        QUESTION: str
        CONTEXTS: list[str]  (abstract sentences)
        LABELS: list[str]  (sentence relevance labels)
        MESHES: list[str]
        LONG_ANSWER: str  (expert long-form answer)
        final_decision: "yes" | "no" | "maybe"

    Output format:
        {id, question, expected_answer_summary, source_ids, domain, difficulty,
         pubmedqa_contexts, pubmedqa_label}
    """
    with open(pubmedqa_path) as f:
        raw = json.load(f)

    qa_pairs = []
    for i, (pmid, entry) in enumerate(raw.items()):
        question = entry.get("QUESTION", "")
        long_answer = entry.get("LONG_ANSWER", "")
        label = entry.get("final_decision", "maybe")
        contexts = entry.get("CONTEXTS", [])
        meshes = entry.get("MESHES", [])

        # Map PubMedQA label to difficulty
        difficulty = "medium" if label == "maybe" else "easy"

        qa_pairs.append({
            "id": f"pqa_{pmid}",
            "question": question,
            "expected_answer_summary": long_answer,
            "source_ids": [f"PMID:{pmid}"],
            "domain": "pubmedqa",
            "difficulty": difficulty,
            "pubmedqa_contexts": contexts,
            "pubmedqa_label": label,
            "pubmedqa_meshes": meshes,
        })

    out = Path(output_path)
    data = {
        "qa_pairs": qa_pairs,
        "metadata": {
            "total_pairs": len(qa_pairs),
            "source": "PubMedQA (PQA-L, expert-labeled)",
            "paper": "Jin et al., EMNLP 2019",
            "label_distribution": {
                "yes": sum(1 for p in qa_pairs if p["pubmedqa_label"] == "yes"),
                "no": sum(1 for p in qa_pairs if p["pubmedqa_label"] == "no"),
                "maybe": sum(1 for p in qa_pairs if p["pubmedqa_label"] == "maybe"),
            },
            "domains": ["pubmedqa"],
            "difficulties": sorted({p["difficulty"] for p in qa_pairs}),
        },
    }

    with open(out, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Converted {len(qa_pairs)} pairs -> {out}")
    label_dist = data["metadata"]["label_distribution"]
    logger.info(f"Labels: yes={label_dist['yes']}, no={label_dist['no']}, maybe={label_dist['maybe']}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Download PubMedQA benchmark dataset")
    parser.add_argument(
        "--output",
        default="data/pubmedqa_labeled.json",
        help="Output path for raw PubMedQA JSON",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Also convert to ClinicalQASet format",
    )
    args = parser.parse_args()

    download_pubmedqa(args.output)

    if args.convert:
        convert_to_qa_format(args.output)


if __name__ == "__main__":
    main()
