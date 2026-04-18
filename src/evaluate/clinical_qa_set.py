"""Clinical Q&A test set loader and manager.

Loads curated clinical Q&A pairs from JSON for evaluation.
Each pair includes a question, expected answer summary, source references,
domain tag, and difficulty level.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ClinicalQASet:
    """Load and manage clinical Q&A test set."""

    def __init__(self, json_path: str | Path = "data/qa_test_set.json"):
        self.json_path = Path(json_path)
        self.qa_pairs: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.json_path.exists():
            return
        with open(self.json_path) as f:
            data = json.load(f)
        self.qa_pairs = data.get("qa_pairs", [])
        self.metadata = data.get("metadata", {})

    def get_all(self) -> list[dict[str, Any]]:
        return list(self.qa_pairs)

    def get_by_id(self, qa_id: str) -> dict[str, Any] | None:
        for pair in self.qa_pairs:
            if pair.get("id") == qa_id:
                return pair
        return None

    def filter_by_domain(self, domain: str) -> list[dict[str, Any]]:
        return [p for p in self.qa_pairs if p.get("domain") == domain]

    def filter_by_difficulty(self, difficulty: str) -> list[dict[str, Any]]:
        return [p for p in self.qa_pairs if p.get("difficulty") == difficulty]

    def get_domains(self) -> list[str]:
        return sorted({p.get("domain", "") for p in self.qa_pairs})

    def get_questions(self) -> list[str]:
        return [p["question"] for p in self.qa_pairs]

    def get_ground_truths(self) -> list[str]:
        return [p.get("expected_answer_summary", "") for p in self.qa_pairs]

    def add_qa_pair(
        self,
        question: str,
        answer: str,
        sources: list[str],
        domain: str,
        difficulty: str = "medium",
        qa_id: str | None = None,
    ) -> None:
        if qa_id is None:
            qa_id = f"qa_{len(self.qa_pairs) + 1:03d}"
        self.qa_pairs.append({
            "id": qa_id,
            "question": question,
            "expected_answer_summary": answer,
            "source_ids": sources,
            "domain": domain,
            "difficulty": difficulty,
        })

    def save(self, path: str | Path | None = None) -> None:
        out_path = Path(path) if path else self.json_path
        data = {
            "qa_pairs": self.qa_pairs,
            "metadata": {
                "total_pairs": len(self.qa_pairs),
                "domains": self.get_domains(),
                "difficulties": sorted({p.get("difficulty", "") for p in self.qa_pairs}),
            },
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    def __len__(self) -> int:
        return len(self.qa_pairs)

    def __repr__(self) -> str:
        return f"ClinicalQASet({len(self)} pairs, domains={self.get_domains()})"
