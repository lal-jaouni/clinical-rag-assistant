"""Tokenizer-free chunking for clinical abstracts.

Chunks text into sentence-level segments, respecting a target token count
(estimated via word count) and overlap between chunks.
"""

from __future__ import annotations


def chunk_abstract(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[dict]:
    """Chunk an abstract into retrievable pieces.

    If text is < 500 tokens, returns a single chunk.
    Otherwise, splits on sentence boundaries (". ") and assembles chunks
    up to chunk_size tokens with overlap tokens of context from the previous chunk.

    Args:
        text: Abstract text to chunk
        chunk_size: Target tokens per chunk
        overlap: Tokens of context to carry from previous chunk

    Returns:
        List of dicts with keys: text, token_count, chunk_index
        Each dict represents one chunk.
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")

    # Estimate token count via word count (simple but effective for clinical text)
    total_tokens = len(text.split())

    # If short, return as single chunk
    if total_tokens < 500:
        return [
            {
                "text": text,
                "token_count": total_tokens,
                "chunk_index": 0,
            }
        ]

    # Split into sentences: on ". " boundary, preserving the period
    sentences = []
    current = []
    for token in text.split():
        current.append(token)
        if token.endswith("."):
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))

    # Greedily assemble chunks with overlap
    chunks = []
    chunk_idx = 0
    sent_idx = 0

    while sent_idx < len(sentences):
        # Start a new chunk; include overlap tokens from the previous chunk if available
        chunk_text = ""
        chunk_token_count = 0
        chunk_sent_count = 0  # Track how many NEW sentences we add (excluding overlap)

        # If not the first chunk, prepend overlap context from previous chunk
        if chunks:
            prev_text = chunks[-1]["text"]
            prev_tokens = prev_text.split()
            overlap_tokens = prev_tokens[-overlap:] if overlap < len(prev_tokens) else prev_tokens
            chunk_text = " ".join(overlap_tokens) + " "
            chunk_token_count = len(overlap_tokens)

        # Accumulate sentences until we reach chunk_size or run out
        while sent_idx < len(sentences):
            sent = sentences[sent_idx]
            sent_token_count = len(sent.split())
            if chunk_token_count + sent_token_count <= chunk_size:
                chunk_text += sent + " "
                chunk_token_count += sent_token_count
                chunk_sent_count += 1
                sent_idx += 1
            else:
                # This sentence would exceed chunk_size
                break

        # If we accumulated at least something (including new sentences), save it
        if chunk_sent_count > 0:
            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "token_count": len(chunk_text.split()),
                    "chunk_index": chunk_idx,
                }
            )
            chunk_idx += 1
        else:
            # Edge case: first sentence of new chunk is longer than chunk_size
            # Force it into a chunk anyway to avoid infinite loop
            if sent_idx < len(sentences):
                sent = sentences[sent_idx]
                chunks.append(
                    {
                        "text": sent,
                        "token_count": len(sent.split()),
                        "chunk_index": chunk_idx,
                    }
                )
                chunk_idx += 1
                sent_idx += 1

    return chunks if chunks else [{"text": text, "token_count": total_tokens, "chunk_index": 0}]


if __name__ == "__main__":
    # Sanity tests
    short = "This is a short abstract. It has only two sentences."
    result_short = chunk_abstract(short)
    print(f"Short abstract test: {len(result_short)} chunk(s)")
    assert len(result_short) == 1, "Short abstract should be 1 chunk"
    assert result_short[0]["chunk_index"] == 0
    print(f"  Text: {result_short[0]['text'][:50]}...")
    print(f"  Tokens: {result_short[0]['token_count']}")

    # Create a long abstract with enough tokens to exceed 500
    long = "This is a word. " * 150  # 150 sentences * 4 words each = 600 tokens
    result_long = chunk_abstract(long, chunk_size=200, overlap=50)
    print(f"\nLong abstract test (~600+ tokens): {len(result_long)} chunk(s)")
    assert len(result_long) > 1, f"Long abstract should be multiple chunks, got {len(result_long)}"
    print(f"  Chunk sizes: {[c['token_count'] for c in result_long]}")
    print(f"  Chunk indices: {[c['chunk_index'] for c in result_long]}")

    print("\nAll sanity tests passed.")
