# Why Open-Source RAG Fails in Clinical Settings (And How to Fix It)

*A deep dive into building retrieval-augmented generation systems that healthcare organizations can actually trust.*

---

Most RAG tutorials end at "retrieve context, generate answer." In clinical settings, that's where the real problems begin.

I spent the last several weeks building a clinical RAG assistant for trauma and critical care -- ingesting PubMed abstracts, FDA guidance documents, and ClinicalTrials.gov data into a retrieval pipeline with safety guardrails. The goal: answer clinical questions with cited evidence and a measurable hallucination rate below 2%.

What I found is that standard open-source RAG architectures have fundamental gaps that make them unsuitable for healthcare without significant additional engineering. This post breaks down those gaps, the solutions I implemented, and what this means for organizations building clinical AI tools.

---

## The Problem: RAG Without Guardrails

A typical RAG system works like this:

1. Embed a question
2. Retrieve similar chunks from a vector store
3. Pass chunks + question to an LLM
4. Return the LLM's answer

This works well enough for customer support chatbots or internal knowledge bases. But clinical settings demand a different standard:

**Hallucination is not an acceptable failure mode.** When an LLM generates a plausible-sounding claim about drug dosing or contraindications that isn't grounded in the retrieved sources, the consequences extend beyond a bad user experience. Standard RAG provides no mechanism to detect or prevent this.

**Source attribution must be verifiable.** Clinicians need to check claims against primary literature. A citation like "[Source 1]" pointing to a chunk ID is useless. They need PMID links, publication years, and enough context to locate the original claim.

**Confidence must be quantified.** Not all questions are equal. A system that answers "What triggers massive transfusion protocol?" with the same certainty as "What is the optimal resuscitation strategy for geriatric polytrauma?" is masking its own uncertainty. Clinicians need to know when the system is guessing.

**Domain boundaries must be enforced.** A clinical RAG system that confidently answers questions about tax law or cooking recipes has no business being deployed in a hospital. Out-of-domain queries need to be detected and refused.

---

## The Solution: Four Layers of Clinical Safety

After evaluating the failure modes, I implemented a multi-layer safety architecture on top of the base RAG pipeline:

### Layer 1: Domain-Specific Retrieval

Generic embedding models perform poorly on biomedical text. Terms like "TXA" (tranexamic acid), "DCR" (damage control resuscitation), and "MTP" (massive transfusion protocol) are either absent from general-purpose embeddings or mapped to incorrect semantic neighborhoods.

The solution combines three techniques:

- **Domain embeddings**: PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`) trained on biomedical literature, replacing general-purpose sentence transformers
- **Hybrid search**: Reciprocal Rank Fusion combining pgvector cosine similarity (weight 0.6) with BM25 keyword matching (weight 0.4) -- because exact term matching matters when "factor V Leiden" and "factor VIII" are clinically different
- **Query expansion**: Automatic abbreviation expansion via a curated synonym map, so "MTP" retrieves chunks mentioning "massive transfusion protocol"

### Layer 2: Source Grounding Verification

After the LLM generates an answer, every sentence is checked against the retrieved source chunks using embedding cosine similarity. If a sentence's maximum similarity to any source chunk falls below 0.50, it is flagged as potentially ungrounded.

This catches the most dangerous hallucination pattern: the LLM synthesizing information that sounds authoritative but isn't present in the retrieved evidence.

A secondary check uses content-word n-gram overlap at the sentence level, catching cases where the embedding similarity is high (semantically similar topic) but the specific claims differ.

### Layer 3: Confidence Scoring

Each answer receives a composite confidence score based on:

- **Citation density**: How many claims in the answer are backed by explicit source references
- **Grounding score**: Average embedding similarity between answer sentences and source chunks
- **Uncertainty language**: Detection of hedging phrases ("may," "possibly," "it is unclear") that signal the LLM is uncertain

When confidence falls below a configurable threshold (default 0.70), the answer is replaced entirely with a safe refusal message directing the user to primary sources. This is a hard override -- the original answer is never shown.

### Layer 4: Direct Advice Detection

Regex-based detection catches prescriptive language patterns: "you should administer," "give the patient X mg," "the recommended dose is." These patterns trigger a safety flag regardless of confidence score, because a retrieval system should surface evidence, not prescribe treatment.

---

## Results: Measuring What Matters

Standard RAG evaluation metrics (answer relevance, context precision) are necessary but insufficient for clinical use. I added clinical-specific metrics:

| Metric | Target | Achieved | Method |
|--------|--------|----------|--------|
| Hallucination rate | < 2% | 0.0% | Sentence-level grounding check against sources |
| Faithfulness | > 0.85 | 0.85+ | Token n-gram overlap between answer and retrieved context |
| Answer relevance | > 0.80 | 0.80+ | Token overlap between answer and original question |
| Context precision | > 0.85 | 0.85+ | Source coverage of expected ground truth |
| Safe refusal rate | Measured | ~15% | Fraction of queries refused due to low confidence |

The 0% hallucination rate on the evaluation set is encouraging but comes with a caveat: the safe refusal mechanism is aggressive. Roughly 15% of queries are refused rather than answered with low confidence. This is the right tradeoff for clinical settings -- it is far better to say "I don't have enough evidence" than to fabricate an answer.

---

## Architecture: What Makes It Work

The full pipeline looks like this:

```
Question
    │
    ▼
Query Processor (abbreviation expansion, domain detection)
    │
    ▼
Hybrid Retriever (pgvector + BM25 via Reciprocal Rank Fusion)
    │
    ▼
Top-K Source Chunks (default 5)
    │
    ▼
LLM Generation (Ollama, local inference, temperature 0.1)
    │
    ▼
Safety Guardrails
    ├── Source grounding check (embedding similarity)
    ├── Hallucination detection (n-gram overlap)
    ├── Confidence scoring (composite)
    ├── Direct advice detection (regex)
    └── Threshold enforcement (refuse or pass)
    │
    ▼
Cited Answer + Source Cards + Confidence Score
```

Key design decisions:

- **Local inference only**: All LLM calls go through Ollama running on the same machine. No patient data or clinical queries leave the network. This is non-negotiable for HIPAA-adjacent deployments.
- **Low temperature (0.1)**: Deterministic generation reduces creative hallucination at the cost of less fluent prose. Clinical accuracy beats readability.
- **Conservative chunking (200 tokens, 50 overlap)**: Smaller chunks mean more precise retrieval at the cost of less context per chunk. The overlap prevents information loss at chunk boundaries.
- **Multiple source types**: PubMed provides research evidence, FDA documents provide regulatory context, and ClinicalTrials.gov provides study-level data. Different source types serve different clinical questions.

---

## What This Means for Healthcare Organizations

The gap between "RAG demo" and "clinically deployable RAG" is substantial. Based on this project, here's where organizations typically need help:

### The Evaluation Gap

Most teams building clinical AI can stand up a RAG pipeline in a week. Few have the infrastructure to measure whether it's safe. Building domain-specific evaluation sets, implementing sentence-level grounding checks, and establishing meaningful safety thresholds requires clinical domain expertise combined with ML engineering.

### The Safety Engineering Gap

Off-the-shelf guardrails (content filters, toxicity detectors) are designed for consumer AI. Clinical safety requires domain-specific mechanisms: grounding verification against medical literature, confidence calibration for clinical questions, and detection of prescriptive language that crosses the line from information to advice.

### The Regulatory Navigation Gap

The FDA's guidance on Clinical Decision Support Software defines four criteria that determine whether a system is regulated as a medical device. Designing a system that stays within the non-device classification while still being clinically useful requires understanding both the technical and regulatory constraints.

### The Deployment Gap

Running a clinical RAG system in production means HIPAA-compliant infrastructure, audit logging, access controls, and monitoring. The ML pipeline is often the easy part; the compliance and operations layer is where most projects stall.

---

## Technical Takeaways

For teams building clinical RAG systems:

1. **Use domain-specific embeddings.** PubMedBERT outperforms general-purpose models on biomedical text by a significant margin. The performance gap is largest on abbreviation-heavy queries.

2. **Hybrid search is not optional.** Pure vector search misses exact-match queries that matter clinically. BM25 catches what embeddings miss, and Reciprocal Rank Fusion combines them cleanly.

3. **Measure hallucination at the sentence level.** Document-level metrics hide sentence-level fabrication. Check every sentence against sources individually.

4. **Refuse rather than hallucinate.** A 15% refusal rate is acceptable. A 2% hallucination rate is not. Calibrate your confidence threshold accordingly.

5. **Keep inference local.** For any system handling clinical queries, local model serving (Ollama, vLLM) eliminates data transmission risk entirely.

6. **Test with adversarial queries.** Include out-of-domain questions, ambiguous abbreviations, and questions that require information your corpus doesn't contain. These edge cases reveal safety gaps that standard benchmarks miss.

---

## What's Next

This project is open source and actively developed. Current focus areas include:

- Expanding the source corpus to additional clinical domains
- Adding real-time PubMed monitoring for new publications
- Implementing user feedback loops for continuous safety calibration
- Building deployment templates for common healthcare infrastructure (AWS GovCloud, Azure for Healthcare)

The broader goal is to establish a reference architecture that healthcare organizations can adapt rather than building clinical safety infrastructure from scratch.

---

*This post describes work on the Clinical RAG Assistant project. The system is designed for clinical decision support and is not a medical device. All evaluation was conducted on published, de-identified data.*
