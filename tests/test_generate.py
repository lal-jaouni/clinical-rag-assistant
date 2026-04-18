"""Tests for generation pipeline: prompts, safety guardrails, output formatting, RAG pipeline.

All tests use mocked LLM responses — no inference needed. Tests validate:
- Prompt template structure and source formatting
- Safety guardrail decision logic (confidence, grounding, advice detection, scope)
- Output formatting with citations, source cards, and metadata
- End-to-end RAG pipeline with mocked retriever and LLM
"""

import re
from unittest.mock import MagicMock, patch

import pytest

from generate.prompt_templates import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    build_messages,
    build_user_prompt,
    format_source_block,
)
from generate.safety_guardrails import (
    SAFE_REFUSAL,
    SafetyGuardrails,
    SafetyResult,
)
from generate.output_formatter import (
    build_citation_link,
    build_source_card,
    format_response,
)
from generate.rag_pipeline import RAGPipeline
from generate.litellm_client import LLMClient, LLMResponse


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures: sample chunks and LLM responses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_CHUNKS = [
    {
        "chunk_id": 1,
        "text": "Massive transfusion protocol is activated when a patient requires "
        "more than 10 units of packed red blood cells within 24 hours.",
        "source_type": "pubmed",
        "source_id": "29451243",
        "title": "MTP Guidelines",
        "year": 2020,
    },
    {
        "chunk_id": 2,
        "text": "Tranexamic acid should be administered within 3 hours of injury "
        "to reduce mortality in hemorrhagic shock patients.",
        "source_type": "pubmed",
        "source_id": "30120987",
        "title": "TXA in Trauma",
        "year": 2021,
    },
    {
        "chunk_id": 3,
        "text": "The FDA requires clinical evaluation for software as a medical "
        "device including machine learning algorithms.",
        "source_type": "fda",
        "source_id": "fda-2021-D-1234",
        "title": "SaMD Clinical Evaluation",
        "year": 2022,
    },
    {
        "chunk_id": 4,
        "text": "Point of care viscoelastic hemostatic assays such as ROTEM and TEG "
        "guide transfusion decisions in massive hemorrhage.",
        "source_type": "clinical_trials",
        "source_id": "NCT04123456",
        "title": "POC-VHA in MTP",
        "year": 2023,
    },
]

WELL_GROUNDED_ANSWER = (
    "Massive transfusion protocol (MTP) is activated when a patient requires "
    "more than 10 units of packed red blood cells within 24 hours [1]. "
    "Tranexamic acid should be administered within 3 hours of injury to reduce "
    "mortality [2]. Viscoelastic assays like ROTEM and TEG can guide ongoing "
    "transfusion decisions [4]."
)

POORLY_GROUNDED_ANSWER = (
    "The optimal blood type for transfusion is O-negative, and all hospitals "
    "should maintain a minimum stock of 500 units at all times. Zinc supplements "
    "have been shown to improve clotting factor synthesis."
)

DIRECT_ADVICE_ANSWER = (
    "You should give the patient 2 units of pRBCs immediately and administer "
    "1g TXA IV over 10 minutes. Start the patient on vasopressors if SBP remains "
    "below 90 mmHg."
)

UNCERTAIN_ANSWER = (
    "Insufficient evidence in the available sources to answer this question."
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt Templates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSystemPrompt:
    def test_contains_citation_rule(self):
        assert "cite" in SYSTEM_PROMPT.lower() or "citation" in SYSTEM_PROMPT.lower()

    def test_contains_refusal_instruction(self):
        assert "insufficient evidence" in SYSTEM_PROMPT.lower() or "i don't know" in SYSTEM_PROMPT.lower()

    def test_prohibits_direct_advice(self):
        assert "never" in SYSTEM_PROMPT.lower() and "patient" in SYSTEM_PROMPT.lower()

    def test_requires_source_only(self):
        assert "only" in SYSTEM_PROMPT.lower() and "source" in SYSTEM_PROMPT.lower()


class TestFormatSourceBlock:
    def test_numbers_sources_sequentially(self):
        block = format_source_block(SAMPLE_CHUNKS[:2])
        assert block.startswith("[1]")
        assert "[2]" in block

    def test_includes_pmid_for_pubmed(self):
        block = format_source_block(SAMPLE_CHUNKS[:1])
        assert "PMID 29451243" in block

    def test_includes_fda_label(self):
        block = format_source_block([SAMPLE_CHUNKS[2]])
        assert "FDA fda-2021-D-1234" in block

    def test_includes_chunk_text(self):
        block = format_source_block(SAMPLE_CHUNKS[:1])
        assert "10 units of packed red blood cells" in block

    def test_includes_title_and_year(self):
        block = format_source_block(SAMPLE_CHUNKS[:1])
        assert "MTP Guidelines" in block
        assert "2020" in block

    def test_empty_chunks(self):
        assert format_source_block([]) == ""

    def test_minimal_chunk_no_metadata(self):
        block = format_source_block([{"text": "Plain text chunk."}])
        assert "[1]" in block
        assert "Plain text chunk." in block


class TestBuildUserPrompt:
    def test_contains_few_shot_examples(self):
        prompt = build_user_prompt("test?", SAMPLE_CHUNKS[:1])
        assert "Example 1" in prompt
        assert "Example 2" in prompt

    def test_contains_question(self):
        prompt = build_user_prompt("What triggers MTP?", SAMPLE_CHUNKS[:1])
        assert "What triggers MTP?" in prompt

    def test_contains_source_block(self):
        prompt = build_user_prompt("test?", SAMPLE_CHUNKS[:2])
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_ends_with_answer_cue(self):
        prompt = build_user_prompt("test?", SAMPLE_CHUNKS[:1])
        assert prompt.rstrip().endswith("Answer:")


class TestBuildMessages:
    def test_returns_system_and_user(self):
        msgs = build_messages("test?", SAMPLE_CHUNKS[:1])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_message_is_system_prompt(self):
        msgs = build_messages("test?", SAMPLE_CHUNKS[:1])
        assert msgs[0]["content"] == SYSTEM_PROMPT


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Safety Guardrails
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSourceGrounding:
    @pytest.fixture
    def guardrails(self):
        return SafetyGuardrails()

    def test_well_grounded_answer_scores_high(self, guardrails):
        score = guardrails.check_source_grounding(WELL_GROUNDED_ANSWER, SAMPLE_CHUNKS)
        assert score > 0.5, f"Expected >0.5, got {score:.3f}"

    def test_poorly_grounded_answer_scores_low(self, guardrails):
        score = guardrails.check_source_grounding(POORLY_GROUNDED_ANSWER, SAMPLE_CHUNKS)
        assert score < 0.3, f"Expected <0.3, got {score:.3f}"

    def test_empty_response_scores_zero(self, guardrails):
        score = guardrails.check_source_grounding("", SAMPLE_CHUNKS)
        assert score == 0.0

    def test_empty_sources_scores_zero(self, guardrails):
        score = guardrails.check_source_grounding("Some answer text", [])
        assert score == 0.0

    def test_exact_copy_scores_high(self, guardrails):
        exact = SAMPLE_CHUNKS[0]["text"]
        score = guardrails.check_source_grounding(exact, SAMPLE_CHUNKS[:1])
        assert score > 0.8, f"Expected >0.8, got {score:.3f}"

    def test_very_short_answer_handled(self, guardrails):
        score = guardrails.check_source_grounding("Yes.", SAMPLE_CHUNKS)
        assert score == 1.0


class TestDirectAdviceDetection:
    @pytest.fixture
    def guardrails(self):
        return SafetyGuardrails()

    def test_detects_give_the_patient(self, guardrails):
        assert guardrails.detect_direct_medical_advice("Give the patient 2 units pRBCs")

    def test_detects_administer_dosage(self, guardrails):
        assert guardrails.detect_direct_medical_advice("Administer 1000 mg TXA IV")

    def test_detects_start_patient_on(self, guardrails):
        assert guardrails.detect_direct_medical_advice("Start the patient on vasopressors")

    def test_detects_you_should_take(self, guardrails):
        assert guardrails.detect_direct_medical_advice("You should take 500 mg aspirin daily")

    def test_no_false_positive_on_literature_report(self, guardrails):
        text = "The study reported that patients who received TXA had lower mortality."
        assert not guardrails.detect_direct_medical_advice(text)

    def test_no_false_positive_on_protocol_description(self, guardrails):
        text = "The protocol recommends administering blood products in a 1:1:1 ratio."
        assert not guardrails.detect_direct_medical_advice(text)

    def test_no_false_positive_on_refusal(self, guardrails):
        assert not guardrails.detect_direct_medical_advice(UNCERTAIN_ANSWER)


class TestConfidenceEstimation:
    @pytest.fixture
    def guardrails(self):
        return SafetyGuardrails()

    def test_uncertain_answer_gets_low_confidence(self, guardrails):
        result = guardrails.validate(UNCERTAIN_ANSWER, SAMPLE_CHUNKS)
        assert result.confidence < 0.5

    def test_well_grounded_cited_answer_gets_high_confidence(self, guardrails):
        result = guardrails.validate(WELL_GROUNDED_ANSWER, SAMPLE_CHUNKS)
        assert result.confidence > 0.6


class TestSafetyValidation:
    @pytest.fixture
    def guardrails(self):
        # No embedding_model in tests → falls back to n-gram grounding.
        # N-gram threshold stays low (0.02) since lexical overlap penalizes paraphrasing.
        return SafetyGuardrails(confidence_threshold=0.7, grounding_threshold=0.02)

    def test_well_grounded_answer_passes(self, guardrails):
        result = guardrails.validate(WELL_GROUNDED_ANSWER, SAMPLE_CHUNKS)
        assert result.is_safe
        assert result.final_response == WELL_GROUNDED_ANSWER

    def test_poorly_grounded_answer_replaced(self, guardrails):
        result = guardrails.validate(POORLY_GROUNDED_ANSWER, SAMPLE_CHUNKS)
        assert not result.is_safe
        assert SAFE_REFUSAL in result.final_response

    def test_direct_advice_flagged(self, guardrails):
        # Use the source chunks that overlap with the advice text so grounding
        # doesn't fire first (we want to isolate the advice detection)
        advice_chunks = [
            {"text": "You should give the patient 2 units of pRBCs immediately and "
             "administer 1g TXA IV over 10 minutes. Start the patient on "
             "vasopressors if SBP remains below 90 mmHg."}
        ]
        result = guardrails.validate(
            DIRECT_ADVICE_ANSWER, advice_chunks, confidence=0.9
        )
        assert result.has_direct_advice
        assert "advice" in result.override_reason

    def test_uncertain_answer_kept_as_is(self, guardrails):
        result = guardrails.validate(UNCERTAIN_ANSWER, SAMPLE_CHUNKS)
        assert result.final_response == UNCERTAIN_ANSWER

    def test_explicit_low_confidence_triggers_refusal(self, guardrails):
        result = guardrails.validate(
            "Some answer without citations.", SAMPLE_CHUNKS, confidence=0.3
        )
        assert not result.is_safe
        assert SAFE_REFUSAL in result.final_response
        assert "confidence" in result.override_reason

    def test_explicit_high_confidence_with_grounding_passes(self, guardrails):
        result = guardrails.validate(
            WELL_GROUNDED_ANSWER, SAMPLE_CHUNKS, confidence=0.9
        )
        assert result.is_safe

    def test_safety_result_has_all_fields(self, guardrails):
        result = guardrails.validate(WELL_GROUNDED_ANSWER, SAMPLE_CHUNKS)
        assert isinstance(result, SafetyResult)
        assert isinstance(result.confidence, float)
        assert isinstance(result.grounding_score, float)
        assert isinstance(result.has_citations, bool)
        assert isinstance(result.has_direct_advice, bool)
        assert isinstance(result.has_uncertainty, bool)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output Formatter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildCitationLink:
    def test_pubmed_citation(self):
        result = build_citation_link({"source_type": "pubmed", "source_id": "12345"})
        assert result["label"] == "[PMID: 12345]"
        assert "pubmed.ncbi.nlm.nih.gov/12345" in result["url"]

    def test_fda_citation(self):
        result = build_citation_link({"source_type": "fda", "source_id": "fda-001"})
        assert result["label"] == "[FDA fda-001]"

    def test_clinical_trials_citation(self):
        result = build_citation_link(
            {"source_type": "clinical_trials", "source_id": "NCT04123456"}
        )
        assert result["label"] == "[CT NCT04123456]"
        assert "clinicaltrials.gov" in result["url"]

    def test_unknown_source_type(self):
        result = build_citation_link({"source_type": "other", "source_id": "abc"})
        assert result["label"] == "[abc]"


class TestBuildSourceCard:
    def test_card_has_required_fields(self):
        card = build_source_card(1, SAMPLE_CHUNKS[0])
        assert card["index"] == 1
        assert "PMID" in card["citation"]
        assert card["title"] == "MTP Guidelines"
        assert card["year"] == 2020
        assert card["source_type"] == "pubmed"
        assert len(card["text_preview"]) > 0

    def test_text_preview_truncated(self):
        long_chunk = {"text": "A " * 500, "source_type": "pubmed", "source_id": "1"}
        card = build_source_card(1, long_chunk)
        assert len(card["text_preview"]) <= 203  # 200 + "..."


class TestFormatResponse:
    def test_basic_structure(self):
        result = format_response(
            answer=WELL_GROUNDED_ANSWER,
            source_chunks=SAMPLE_CHUNKS,
            confidence=0.85,
            grounding_score=0.72,
            latency_ms=1500,
            model="ollama/llama3.1:8b",
        )
        assert result["answer"] == WELL_GROUNDED_ANSWER
        assert result["confidence"] == 0.85
        assert result["grounding_score"] == 0.72
        assert result["is_safe"] is True
        assert result["latency_ms"] == 1500
        assert result["model"] == "ollama/llama3.1:8b"

    def test_sources_list_matches_chunks(self):
        result = format_response(
            answer="test", source_chunks=SAMPLE_CHUNKS,
            confidence=0.8, grounding_score=0.7,
        )
        assert len(result["sources"]) == len(SAMPLE_CHUNKS)

    def test_cited_indices_extracted(self):
        answer = "Something [1] and also [3] and [1] again."
        result = format_response(
            answer=answer, source_chunks=SAMPLE_CHUNKS,
            confidence=0.8, grounding_score=0.7,
        )
        assert result["cited_indices"] == [1, 3]

    def test_no_citations_empty_list(self):
        result = format_response(
            answer="No brackets here.", source_chunks=SAMPLE_CHUNKS,
            confidence=0.8, grounding_score=0.7,
        )
        assert result["cited_indices"] == []

    def test_override_reason_propagated(self):
        result = format_response(
            answer=SAFE_REFUSAL, source_chunks=[],
            confidence=0.0, grounding_score=0.0,
            is_safe=False, override_reason="no sources retrieved",
        )
        assert result["is_safe"] is False
        assert result["override_reason"] == "no sources retrieved"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG Pipeline (mocked LLM + retriever)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _mock_llm_response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="ollama/llama3.1:8b",
        provider="ollama",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=800,
        raw=None,
    )


class TestRAGPipeline:
    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = SAMPLE_CHUNKS
        return retriever

    @pytest.fixture
    def mock_embedding(self):
        em = MagicMock()
        em.embed_query.return_value = [0.1] * 768
        return em

    @pytest.fixture
    def pipeline(self, mock_retriever, mock_embedding):
        llm = MagicMock(spec=LLMClient)
        llm.model = "ollama/llama3.1:8b"
        llm.complete.return_value = _mock_llm_response(WELL_GROUNDED_ANSWER)

        return RAGPipeline(
            llm=llm,
            retriever=mock_retriever,
            guardrails=SafetyGuardrails(),
            embedding_model=mock_embedding,
        )

    def test_returns_structured_response(self, pipeline):
        result = pipeline.answer("What triggers MTP?")
        assert "answer" in result
        assert "confidence" in result
        assert "sources" in result
        assert "latency_ms" in result

    def test_calls_retriever(self, pipeline, mock_retriever):
        pipeline.answer("test query")
        mock_retriever.retrieve.assert_called_once()

    def test_calls_llm(self, pipeline):
        pipeline.answer("test query")
        pipeline.llm.complete.assert_called_once()

    def test_well_grounded_answer_passes_through(self, pipeline):
        result = pipeline.answer("What triggers MTP?")
        assert result["is_safe"]
        assert WELL_GROUNDED_ANSWER in result["answer"]

    def test_poorly_grounded_answer_replaced(self, pipeline):
        pipeline.llm.complete.return_value = _mock_llm_response(POORLY_GROUNDED_ANSWER)
        result = pipeline.answer("test?")
        assert not result["is_safe"]
        assert SAFE_REFUSAL in result["answer"]

    def test_empty_retrieval_returns_safe_refusal(self, pipeline, mock_retriever):
        mock_retriever.retrieve.return_value = []
        result = pipeline.answer("obscure query?")
        assert "No relevant sources" in result["answer"]

    def test_direct_advice_flagged(self, pipeline, mock_retriever):
        # Return chunks that overlap with the advice text so grounding passes
        advice_chunks = [
            {"chunk_id": 1, "text": DIRECT_ADVICE_ANSWER,
             "source_type": "pubmed", "source_id": "99999", "title": "Test", "year": 2024}
        ]
        mock_retriever.retrieve.return_value = advice_chunks
        pipeline.llm.complete.return_value = _mock_llm_response(DIRECT_ADVICE_ANSWER)
        pipeline.guardrails.confidence_threshold = 0.0
        pipeline.guardrails.grounding_threshold = 0.0
        result = pipeline.answer("test?")
        assert result["override_reason"] is not None
        assert "advice" in result["override_reason"]

    def test_query_processor_called_when_present(self, pipeline):
        qp = MagicMock()
        qp.process.return_value = {"expanded": "expanded query", "original": "q", "synonyms_found": {}}
        pipeline.query_processor = qp
        pipeline.answer("q")
        qp.process.assert_called_once_with("q")

    def test_latency_is_positive(self, pipeline):
        result = pipeline.answer("test?")
        assert result["latency_ms"] >= 0

    def test_model_name_in_response(self, pipeline):
        result = pipeline.answer("test?")
        assert result["model"] == "ollama/llama3.1:8b"

    def test_sources_included_in_response(self, pipeline):
        result = pipeline.answer("test?")
        assert len(result["sources"]) == len(SAMPLE_CHUNKS)

    def test_cited_indices_extracted(self, pipeline):
        result = pipeline.answer("test?")
        # WELL_GROUNDED_ANSWER cites [1], [2], [4]
        assert 1 in result["cited_indices"]
        assert 2 in result["cited_indices"]
        assert 4 in result["cited_indices"]
