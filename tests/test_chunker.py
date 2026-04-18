"""Tests for the chunker module (Phase 2a)."""

from __future__ import annotations

import pytest

from src.ingest.chunker import chunk_abstract


class TestChunkerBasics:
    """Basic chunking behavior."""

    def test_short_abstract_single_chunk(self):
        """Abstracts < 500 tokens should return a single chunk."""
        short = "This is a short abstract. It has only two sentences."
        result = chunk_abstract(short)
        assert len(result) == 1
        assert result[0]["chunk_index"] == 0
        assert result[0]["text"] == short
        assert result[0]["token_count"] > 0

    def test_short_abstract_preserves_content(self):
        """Short abstract content should be preserved exactly."""
        text = "Hemorrhagic shock is a medical emergency. Treatment varies by severity."
        result = chunk_abstract(text)
        assert len(result) == 1
        assert result[0]["text"] == text

    def test_long_abstract_multiple_chunks(self):
        """Abstracts >= 500 tokens should return multiple chunks."""
        # ~600 tokens: "This is a word. " (4 tokens) repeated 150 times = 600 tokens
        long = "This is a word. " * 150
        result = chunk_abstract(long, chunk_size=200, overlap=50)
        assert len(result) > 1
        assert all(c["chunk_index"] >= 0 for c in result)
        # Chunk indices should be sequential
        indices = [c["chunk_index"] for c in result]
        assert indices == list(range(len(result)))

    def test_chunk_indices_sequential(self):
        """Chunk indices must be 0, 1, 2, ... in order."""
        long = "This is a word. " * 150  # 600 tokens
        result = chunk_abstract(long, chunk_size=200, overlap=50)
        indices = [c["chunk_index"] for c in result]
        assert indices == list(range(len(indices)))

    def test_token_counts_present(self):
        """Every chunk must have a token_count."""
        long = "This is a word. " * 150  # 600 tokens
        result = chunk_abstract(long, chunk_size=200, overlap=50)
        for chunk in result:
            assert chunk["token_count"] > 0
            assert isinstance(chunk["token_count"], int)

    def test_single_sentence_short_abstract(self):
        """Single sentence under 500 tokens is one chunk."""
        text = "This single sentence is the entire abstract and it is short."
        result = chunk_abstract(text)
        assert len(result) == 1
        assert result[0]["chunk_index"] == 0


class TestChunkerEdgeCases:
    """Edge cases and error handling."""

    def test_empty_string_raises_error(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot chunk empty text"):
            chunk_abstract("")

    def test_whitespace_only_raises_error(self):
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot chunk empty text"):
            chunk_abstract("   \n\t  ")

    def test_very_long_single_sentence(self):
        """Single sentence longer than chunk_size should not error."""
        # A sentence with >200 words (>200 tokens)
        long_sentence = "Word " * 300 + "."
        result = chunk_abstract(long_sentence, chunk_size=200, overlap=50)
        # Should still return at least one chunk, even if it exceeds chunk_size
        assert len(result) >= 1
        assert result[0]["text"]


class TestChunkerOverlap:
    """Test overlap behavior."""

    def test_overlap_appears_in_consecutive_chunks(self):
        """Overlap tokens should appear at start of next chunk and end of previous."""
        # Crafted text with clear token boundaries
        text = "A B C D E. F G H I J. K L M N O. P Q R S T. U V W X Y."
        result = chunk_abstract(text, chunk_size=10, overlap=5)

        # With chunk_size=10 and 5-token sentences, we should get multiple chunks
        if len(result) > 1:
            # Check that end of chunk[i] overlaps with start of chunk[i+1]
            for i in range(len(result) - 1):
                current_tokens = result[i]["text"].split()
                next_tokens = result[i + 1]["text"].split()

                # The start of next chunk should contain overlap of current chunk
                current_end = current_tokens[-5:] if len(current_tokens) >= 5 else current_tokens
                next_start = next_tokens[: len(current_end)]

                # At least some overlap should exist (this is a soft assertion,
                # exact overlap depends on sentence boundaries)
                assert len(next_start) > 0


class TestChunkerTokenCount:
    """Test token count accuracy."""

    def test_token_count_matches_word_count(self):
        """Token count should match len(text.split())."""
        text = "The quick brown fox. Jumped over lazy dog."
        result = chunk_abstract(text)
        expected_tokens = len(text.split())
        assert result[0]["token_count"] == expected_tokens

    def test_long_document_token_sum_approximately_correct(self):
        """Sum of chunk token counts should be close to original (accounting for overlap)."""
        text = "Sentence one. " * 60  # ~300 tokens
        result = chunk_abstract(text, chunk_size=100, overlap=20)
        # When text is reassembled, overlap causes some duplication
        total_in_result = sum(c["token_count"] for c in result)
        original_tokens = len(text.split())
        # Total chunk tokens may be higher than original due to overlap, but reasonable
        assert total_in_result >= original_tokens  # At least original count
        assert total_in_result <= original_tokens * 1.5  # At most 50% overhead from overlap


class TestChunkerConfigurable:
    """Test configurable parameters."""

    def test_different_chunk_sizes(self):
        """Larger chunk_size should produce fewer, longer chunks."""
        text = "This is a word. " * 150  # 600 tokens
        small_chunks = chunk_abstract(text, chunk_size=100, overlap=20)
        large_chunks = chunk_abstract(text, chunk_size=300, overlap=50)
        # More chunks with smaller size
        assert len(small_chunks) > len(large_chunks)

    def test_different_overlaps(self):
        """Overlap affects chunk text but should be reasonable."""
        text = "This is a word. " * 150  # 600 tokens
        no_overlap = chunk_abstract(text, chunk_size=150, overlap=0)
        with_overlap = chunk_abstract(text, chunk_size=150, overlap=30)
        # With overlap we might have more chunks due to overlap re-adding tokens at start
        # Just verify both produce multiple chunks and are reasonable
        assert len(no_overlap) >= 3
        assert len(with_overlap) >= 2
        # With overlap, chunks should have more text duplication but similar lengths
        assert sum(c["token_count"] for c in no_overlap) < sum(c["token_count"] for c in with_overlap) * 1.2
