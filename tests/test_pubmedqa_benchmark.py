"""Tests for PubMedQA benchmark evaluation."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.evaluate.pubmedqa_benchmark import _classify_answer, run_pubmedqa_benchmark


class TestClassifyAnswer:
    """Test yes/no/maybe classification from generated text."""

    def test_explicit_yes(self):
        assert _classify_answer("Yes, the study confirms this.") == "yes"

    def test_explicit_no(self):
        assert _classify_answer("No, the evidence does not support this.") == "no"

    def test_explicit_maybe(self):
        assert _classify_answer("Maybe, the evidence is mixed.") == "maybe"

    def test_the_answer_is_yes(self):
        assert _classify_answer("The answer is yes based on the evidence.") == "yes"

    def test_the_answer_is_no(self):
        assert _classify_answer("The answer is no, studies show otherwise.") == "no"

    def test_the_answer_is_maybe(self):
        assert _classify_answer("The answer is maybe due to limited data.") == "maybe"

    def test_signal_based_yes(self):
        assert _classify_answer("The studies confirm that the treatment works. Evidence supports efficacy.") == "yes"

    def test_signal_based_no(self):
        assert _classify_answer("There was no significant difference. The study did not show improvement.") == "no"

    def test_signal_based_maybe(self):
        assert _classify_answer("The results are inconclusive. Further research is needed.") == "maybe"

    def test_ambiguous_defaults_to_maybe(self):
        assert _classify_answer("The protein interacts with the receptor.") == "maybe"

    def test_empty_string_defaults_to_maybe(self):
        assert _classify_answer("") == "maybe"

    def test_case_insensitive(self):
        assert _classify_answer("YES, the findings are clear.") == "yes"
        assert _classify_answer("NO, this is incorrect.") == "no"

    def test_multiline_first_line_priority(self):
        answer = "Yes, the evidence supports this.\nHowever, no significant effect was found."
        assert _classify_answer(answer) == "yes"


class TestDownloadScript:
    """Test download and conversion scripts."""

    def test_converted_file_format(self, tmp_path):
        """Test that converted file has correct structure."""
        raw = {
            "12345": {
                "QUESTION": "Does drug X work?",
                "CONTEXTS": ["Context sentence 1.", "Context sentence 2."],
                "LABELS": ["BACKGROUND", "RESULTS"],
                "MESHES": ["Humans", "Drug"],
                "LONG_ANSWER": "The drug shows promise in trials.",
                "final_decision": "yes",
            }
        }
        raw_path = tmp_path / "raw.json"
        raw_path.write_text(json.dumps(raw))

        from scripts.download_pubmedqa import convert_to_qa_format

        out_path = tmp_path / "converted.json"
        convert_to_qa_format(str(raw_path), str(out_path))

        with open(out_path) as f:
            data = json.load(f)

        assert "qa_pairs" in data
        assert "metadata" in data
        assert len(data["qa_pairs"]) == 1

        pair = data["qa_pairs"][0]
        assert pair["id"] == "pqa_12345"
        assert pair["question"] == "Does drug X work?"
        assert pair["pubmedqa_label"] == "yes"
        assert pair["domain"] == "pubmedqa"
        assert len(pair["pubmedqa_contexts"]) == 2
        assert pair["difficulty"] == "easy"  # yes -> easy

    def test_maybe_label_maps_to_medium(self, tmp_path):
        raw = {
            "99999": {
                "QUESTION": "Is it unclear?",
                "CONTEXTS": [],
                "LABELS": [],
                "MESHES": [],
                "LONG_ANSWER": "Unknown.",
                "final_decision": "maybe",
            }
        }
        raw_path = tmp_path / "raw.json"
        raw_path.write_text(json.dumps(raw))

        from scripts.download_pubmedqa import convert_to_qa_format

        out_path = tmp_path / "converted.json"
        convert_to_qa_format(str(raw_path), str(out_path))

        with open(out_path) as f:
            data = json.load(f)

        assert data["qa_pairs"][0]["difficulty"] == "medium"

    def test_label_distribution_in_metadata(self, tmp_path):
        raw = {
            "1": {"QUESTION": "Q1", "CONTEXTS": [], "LABELS": [], "MESHES": [],
                  "LONG_ANSWER": "A1", "final_decision": "yes"},
            "2": {"QUESTION": "Q2", "CONTEXTS": [], "LABELS": [], "MESHES": [],
                  "LONG_ANSWER": "A2", "final_decision": "no"},
            "3": {"QUESTION": "Q3", "CONTEXTS": [], "LABELS": [], "MESHES": [],
                  "LONG_ANSWER": "A3", "final_decision": "maybe"},
        }
        raw_path = tmp_path / "raw.json"
        raw_path.write_text(json.dumps(raw))

        from scripts.download_pubmedqa import convert_to_qa_format

        out_path = tmp_path / "converted.json"
        convert_to_qa_format(str(raw_path), str(out_path))

        with open(out_path) as f:
            data = json.load(f)

        dist = data["metadata"]["label_distribution"]
        assert dist["yes"] == 1
        assert dist["no"] == 1
        assert dist["maybe"] == 1


class TestRunBenchmark:
    """Test benchmark runner with mocked LLM."""

    @pytest.fixture
    def mini_dataset(self, tmp_path):
        """Create a minimal PubMedQA test set."""
        data = {
            "qa_pairs": [
                {
                    "id": "pqa_test_1",
                    "question": "Does aspirin reduce inflammation?",
                    "expected_answer_summary": "Yes, aspirin inhibits COX enzymes.",
                    "source_ids": ["PMID:1"],
                    "domain": "pubmedqa",
                    "difficulty": "easy",
                    "pubmedqa_contexts": [
                        "Aspirin inhibits cyclooxygenase enzymes.",
                        "This reduces prostaglandin synthesis and inflammation.",
                    ],
                    "pubmedqa_label": "yes",
                    "pubmedqa_meshes": ["Aspirin"],
                },
                {
                    "id": "pqa_test_2",
                    "question": "Is homeopathy effective for cancer?",
                    "expected_answer_summary": "No evidence supports homeopathy.",
                    "source_ids": ["PMID:2"],
                    "domain": "pubmedqa",
                    "difficulty": "easy",
                    "pubmedqa_contexts": [
                        "No randomized trials support homeopathy for cancer.",
                    ],
                    "pubmedqa_label": "no",
                    "pubmedqa_meshes": ["Homeopathy"],
                },
            ],
            "metadata": {
                "total_pairs": 2,
                "source": "test",
                "label_distribution": {"yes": 1, "no": 1, "maybe": 0},
            },
        }
        path = tmp_path / "test_set.json"
        path.write_text(json.dumps(data))
        return path

    @patch("src.evaluate.pubmedqa_benchmark._build_llm_client")
    @patch("src.evaluate.pubmedqa_benchmark._build_safety")
    def test_benchmark_returns_report(self, mock_safety, mock_llm, mini_dataset, tmp_path):
        """Test that benchmark produces a valid report structure."""
        # Mock LLM
        mock_response = MagicMock()
        mock_response.text = "Yes, aspirin reduces inflammation by inhibiting COX."
        mock_client = MagicMock()
        mock_client.complete.return_value = mock_response
        mock_llm.return_value = mock_client

        # Mock safety (returns SafetyResult dataclass)
        mock_guard = MagicMock()
        mock_result = MagicMock()
        mock_result.final_response = "Yes, aspirin reduces inflammation by inhibiting COX."
        mock_result.confidence = 0.9
        mock_result.grounding_score = 0.85
        mock_result.is_safe = True
        mock_result.override_reason = None
        mock_guard.validate.return_value = mock_result
        mock_safety.return_value = mock_guard

        report = run_pubmedqa_benchmark(
            qa_path=str(mini_dataset),
            output_dir=str(tmp_path / "output"),
            limit=2,
        )

        assert report["benchmark"] == "PubMedQA (PQA-L, expert-labeled)"
        assert report["mode"] == "generation_only"
        assert report["num_questions"] == 2
        assert report["num_answered"] == 2
        assert "accuracy" in report
        assert "ragas" in report
        assert "hallucination_rate" in report
        assert 0 <= report["accuracy"]["overall"] <= 1

        # Verify report was saved
        assert (tmp_path / "output" / "pubmedqa_report.json").exists()
