"""Query preprocessing and clinical synonym expansion.

Normalizes clinical queries and optionally expands abbreviations/synonyms
so retrieval catches more relevant documents.
"""

from __future__ import annotations

from typing import Any

# Clinical synonym map: abbreviation/shorthand -> expanded forms
# Used to augment BM25 keyword matching (vector search handles semantic similarity)
DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "MTP": ["massive transfusion protocol", "massive transfusion"],
    "MHP": ["massive hemorrhage protocol"],
    "TXA": ["tranexamic acid"],
    "DCR": ["damage control resuscitation"],
    "FFP": ["fresh frozen plasma"],
    "PRBCs": ["packed red blood cells"],
    "RBCs": ["red blood cells"],
    "PLTs": ["platelets"],
    "ROTEM": ["rotational thromboelastometry"],
    "TEG": ["thromboelastography"],
    "INR": ["international normalized ratio"],
    "aPTT": ["activated partial thromboplastin time"],
    "DIC": ["disseminated intravascular coagulation"],
    "CRASH-2": ["clinical randomisation of an antifibrinolytic in significant haemorrhage"],
    "PROPPR": ["pragmatic randomized optimal platelet and plasma ratios"],
    "SBP": ["systolic blood pressure"],
    "MAP": ["mean arterial pressure"],
    "GCS": ["glasgow coma scale"],
    "ISS": ["injury severity score"],
    "ED": ["emergency department"],
    "ICU": ["intensive care unit"],
    "ATLS": ["advanced trauma life support"],
    "POC": ["point of care"],
    "VHA": ["viscoelastic hemostatic assay", "viscoelastic hemostatic assays"],
    "ABC": ["assessment of blood consumption"],
    "TASH": ["trauma associated severe hemorrhage"],
    "SI": ["shock index"],
}


class QueryProcessor:
    """Preprocess queries for retrieval optimization.

    Handles synonym expansion and basic normalization for clinical terminology.
    """

    def __init__(self, synonym_map: dict[str, list[str]] | None = None):
        self.synonym_map = synonym_map if synonym_map is not None else DEFAULT_SYNONYMS

    def process(self, query: str) -> dict[str, Any]:
        """Process query for retrieval.

        Args:
            query: User query string

        Returns:
            Dict with: original, expanded, synonyms_found
        """
        expanded_terms = []
        synonyms_found = {}

        words = query.split()
        for word in words:
            clean = word.strip(".,?!;:()").upper()
            if clean in self.synonym_map:
                expansions = self.synonym_map[clean]
                synonyms_found[clean] = expansions
                expanded_terms.extend(expansions)

        # Build expanded query: original + synonym expansions
        expanded = query
        if expanded_terms:
            expanded = query + " " + " ".join(expanded_terms)

        return {
            "original": query,
            "expanded": expanded,
            "synonyms_found": synonyms_found,
        }
