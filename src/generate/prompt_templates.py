"""Clinical-specific prompt templates."""

from typing import List, Dict, Any


class PromptTemplates:
    """Clinical RAG prompt templates with safety guardrails."""

    @staticmethod
    def system_prompt() -> str:
        """System prompt for clinical safety."""
        return """You are a clinical research assistant helping healthcare professionals find evidence-based information.

Your responsibilities:
1. Answer clinical questions based ONLY on provided source documents
2. Always cite the source (PubMed ID, FDA document, etc.) for every fact
3. Never provide direct medical advice to patients
4. If you are uncertain, say "I don't know" rather than guessing
5. Highlight any limitations or conflicts in the evidence
6. Keep responses concise and actionable for clinicians

Remember: You support clinical decision-making, not clinical practice."""

    @staticmethod
    def query_prompt(query: str, context: List[Dict[str, Any]]) -> str:
        """Generate prompt for a clinical query with context.

        Args:
            query: Clinical question
            context: List of retrieved document chunks

        Returns:
            Formatted prompt string
        """
        # Implementation placeholder
        pass

    @staticmethod
    def few_shot_examples() -> str:
        """Few-shot examples for clinical Q&A with citations."""
        return """Example 1:
Q: What is the mortality rate in hemorrhagic shock?
A: Mortality in hemorrhagic shock varies by severity. Class III hemorrhage (loss of 30-40% blood volume) has estimated mortality of 20-40%, while Class IV (>40% loss) approaches 50-100% without rapid intervention [PMID: 12345678]. Early recognition and source control are critical.

Example 2:
Q: Can I give adrenaline to a patient in cardiac arrest?
A: I cannot provide direct medical advice to patients. For cardiac arrest management, refer to current ACLS guidelines and your institution's protocols. The research literature supports epinephrine in specific scenarios [PMID: 87654321], but dosing and timing must follow established protocols."""
