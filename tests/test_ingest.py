"""Tests for document ingestion pipeline."""

import pytest
from src.ingest.chunker import ClinicalChunker
from src.ingest.pubmed_loader import PubMedLoader


class TestClinicalChunker:
    """Test clinical-aware chunking."""

    @pytest.fixture
    def chunker(self):
        return ClinicalChunker(chunk_size=200, overlap=50)

    @pytest.fixture
    def sample_text(self):
        return """
        Hemorrhagic shock is a state of acute circulatory failure resulting from a loss of circulating blood volume.
        Class I hemorrhage involves loss of up to 15% of blood volume.
        Class II involves 15-30% blood loss.
        Class III involves 30-40% blood loss with mortality of 20-40%.
        Class IV involves >40% blood loss with mortality approaching 50-100%.
        """

    def test_chunk_preserves_metadata(self, chunker, sample_text):
        """Test that chunking preserves document metadata."""
        metadata = {
            "source": "pubmed",
            "pmid": "12345678",
            "doi": "10.1234/example",
            "date": "2023-01-01",
        }
        result = chunker.chunk(sample_text, metadata)
        assert all("source" in chunk["metadata"] for chunk in result)
        assert all(chunk["metadata"]["source"] == "pubmed" for chunk in result)

    def test_chunk_overlap(self, chunker, sample_text):
        """Test that chunks have configured overlap."""
        # Placeholder test
        pass


class TestPubMedLoader:
    """Test PubMed ingestion."""

    @pytest.fixture
    def loader(self):
        return PubMedLoader(mesh_terms=["Hemorrhagic Shock"])

    def test_validate_config(self, loader):
        """Test config validation."""
        assert loader.validate_config()

    def test_load_batch_size(self, loader):
        """Test batch size configuration."""
        # Placeholder test
        pass
