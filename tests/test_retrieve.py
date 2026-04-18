"""Tests for retrieval pipeline (BM25, hybrid fusion, reranker, query processor).

Vector store tests require a running PostgreSQL instance and are skipped
in CI. BM25, RRF, and query processing are tested with in-memory data.
"""

import pytest

from src.retrieve.hybrid_retriever import BM25Index, HybridRetriever, reciprocal_rank_fusion
from src.retrieve.query_processor import QueryProcessor, DEFAULT_SYNONYMS
from src.retrieve.reranker import Reranker


# ── Sample corpus for BM25 tests ──────────────────────────────────────

SAMPLE_CORPUS = [
    {
        "chunk_id": 1,
        "document_id": 10,
        "text": "Massive transfusion protocol is activated when a patient requires more than 10 units of packed red blood cells within 24 hours.",
        "source_type": "pubmed",
        "source_id": "12345",
        "title": "MTP Guidelines",
        "year": 2020,
    },
    {
        "chunk_id": 2,
        "document_id": 11,
        "text": "Tranexamic acid should be administered within 3 hours of injury to reduce mortality in hemorrhagic shock patients.",
        "source_type": "pubmed",
        "source_id": "12346",
        "title": "TXA in Trauma",
        "year": 2021,
    },
    {
        "chunk_id": 3,
        "document_id": 12,
        "text": "Damage control resuscitation emphasizes permissive hypotension, hemostatic resuscitation, and early surgical intervention.",
        "source_type": "pubmed",
        "source_id": "12347",
        "title": "DCR Principles",
        "year": 2019,
    },
    {
        "chunk_id": 4,
        "document_id": 13,
        "text": "The FDA requires clinical evaluation for software as a medical device including machine learning algorithms used in clinical decision support.",
        "source_type": "fda",
        "source_id": "fda-001",
        "title": "SaMD Clinical Evaluation",
        "year": 2022,
    },
    {
        "chunk_id": 5,
        "document_id": 14,
        "text": "Point of care viscoelastic hemostatic assays such as ROTEM and TEG guide transfusion decisions in massive hemorrhage.",
        "source_type": "pubmed",
        "source_id": "12348",
        "title": "POC-VHA in MTP",
        "year": 2023,
    },
]


class TestBM25Index:
    @pytest.fixture
    def bm25(self):
        return BM25Index(SAMPLE_CORPUS)

    def test_search_returns_results(self, bm25):
        results = bm25.search("massive transfusion protocol", top_k=3)
        assert len(results) <= 3
        assert all("bm25_score" in r for r in results)

    def test_mtp_query_ranks_mtp_doc_first(self, bm25):
        results = bm25.search("massive transfusion protocol activation criteria", top_k=3)
        assert results[0]["chunk_id"] == 1

    def test_txa_query_ranks_txa_doc_first(self, bm25):
        results = bm25.search("tranexamic acid hemorrhagic shock mortality", top_k=3)
        assert results[0]["chunk_id"] == 2

    def test_fda_query_finds_fda_doc(self, bm25):
        results = bm25.search("FDA clinical evaluation software medical device", top_k=3)
        assert any(r["chunk_id"] == 4 for r in results)

    def test_top_k_limit(self, bm25):
        results = bm25.search("transfusion", top_k=2)
        assert len(results) == 2


class TestReciprocalRankFusion:
    def test_rrf_merges_two_lists(self):
        list_a = [{"chunk_id": 1, "text": "a"}, {"chunk_id": 2, "text": "b"}]
        list_b = [{"chunk_id": 2, "text": "b"}, {"chunk_id": 3, "text": "c"}]

        fused = reciprocal_rank_fusion([list_a, list_b])
        ids = [r["chunk_id"] for r in fused]

        # chunk_id 2 appears in both lists, should rank highest
        assert ids[0] == 2
        assert set(ids) == {1, 2, 3}

    def test_rrf_single_list(self):
        items = [{"chunk_id": 10, "text": "x"}, {"chunk_id": 20, "text": "y"}]
        fused = reciprocal_rank_fusion([items])
        assert len(fused) == 2
        assert fused[0]["chunk_id"] == 10  # rank 1 > rank 2

    def test_rrf_empty(self):
        fused = reciprocal_rank_fusion([])
        assert fused == []

    def test_rrf_scores_are_floats(self):
        items = [{"chunk_id": 1, "text": "a"}]
        fused = reciprocal_rank_fusion([items])
        assert isinstance(fused[0]["rrf_score"], float)


class TestHybridRetriever:
    def test_weight_configuration(self):
        retriever = HybridRetriever(vector_store=None, vector_weight=0.7, bm25_weight=0.3)
        assert retriever.vector_weight == 0.7
        assert retriever.bm25_weight == 0.3


class TestQueryProcessor:
    @pytest.fixture
    def processor(self):
        return QueryProcessor()

    def test_expands_known_abbreviation(self, processor):
        result = processor.process("What is the MTP activation criteria?")
        assert "massive transfusion protocol" in result["expanded"].lower()
        assert "MTP" in result["synonyms_found"]

    def test_expands_multiple_abbreviations(self, processor):
        result = processor.process("Give TXA in DCR")
        assert "TXA" in result["synonyms_found"]
        assert "DCR" in result["synonyms_found"]

    def test_preserves_original(self, processor):
        query = "What is hemorrhagic shock?"
        result = processor.process(query)
        assert result["original"] == query

    def test_no_expansion_for_unknown_terms(self, processor):
        result = processor.process("regular query with no abbreviations")
        assert result["expanded"] == result["original"]
        assert result["synonyms_found"] == {}

    def test_custom_synonym_map(self):
        custom = {"HTN": ["hypertension"]}
        processor = QueryProcessor(synonym_map=custom)
        result = processor.process("Does HTN affect outcomes?")
        assert "hypertension" in result["expanded"]

    def test_default_synonyms_coverage(self):
        assert len(DEFAULT_SYNONYMS) >= 20


class TestReranker:
    def test_init(self):
        reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def test_rerank_empty_documents(self):
        reranker = Reranker()
        result = reranker.rerank("test query", [])
        assert result == []
