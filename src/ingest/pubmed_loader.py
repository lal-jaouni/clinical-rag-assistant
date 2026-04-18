"""PubMed document loader via NCBI Entrez API."""

from __future__ import annotations

import logging
import socket
import time
from typing import Any

from Bio import Entrez

from ingest.base import BaseDocumentLoader

logger = logging.getLogger(__name__)


# NCBI Entrez resolves AAAA records, but some networks (incl. this VM) have broken
# IPv6 egress. Force IPv4 resolution to avoid 10s SYN-SENT hangs on every request.
_original_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, *args, **kwargs):
    return _original_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)


socket.getaddrinfo = _ipv4_only_getaddrinfo


class PubMedLoader(BaseDocumentLoader):
    """Fetch PubMed abstracts via NCBI Entrez API.

    Retrieves articles by MeSH terms, extracts metadata (PMID, DOI, date, authors),
    and returns structured documents with abstracts.
    """

    def __init__(
        self,
        api_key: str = "",
        email: str = "",
        mesh_terms: list[str] | None = None,
        max_per_term: int = 100,
        date_range_start: int = 2010,
    ):
        """Initialize PubMed loader.

        Args:
            api_key: NCBI API key (optional, for higher rate limits)
            email: Email for NCBI Entrez (required by NCBI)
            mesh_terms: MeSH terms to filter
            max_per_term: Max results per MeSH term
            date_range_start: Minimum publication year
        """
        self.api_key = api_key
        self.email = email or "laith@personalagent.local"
        self.mesh_terms = mesh_terms or []
        self.max_per_term = max_per_term
        self.date_range_start = date_range_start
        self.rate_limit_delay = 0.34 if not api_key else 0.1  # ~3 req/s without API key

    def validate_config(self) -> bool:
        """Validate Entrez API availability."""
        if not self.email:
            raise ValueError("Entrez email not set")
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key
        return True

    def load(
        self,
        batch_size: int = 100,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch PubMed abstracts.

        Args:
            batch_size: Batch size for efetch (up to 100)
            max_results: Ignored; uses per-term limits instead

        Returns:
            List of document dicts with pmid, title, authors, year, abstract, doi, etc.
        """
        self.validate_config()

        # Phase 1: search for all PMIDs across terms
        all_pmids = set()
        for mesh_term in self.mesh_terms:
            pmids_for_term = self._search_term(mesh_term, self.max_per_term)
            all_pmids.update(pmids_for_term)
            time.sleep(self.rate_limit_delay)

        logger.info(f"Found {len(all_pmids)} unique PMIDs across {len(self.mesh_terms)} terms")

        # Phase 2: fetch full records for each PMID in batches
        pmid_list = sorted(all_pmids)
        documents = []

        for i in range(0, len(pmid_list), batch_size):
            batch_pmids = pmid_list[i : i + batch_size]
            batch_docs = self._fetch_batch(batch_pmids)
            documents.extend(batch_docs)
            if i + batch_size < len(pmid_list):
                time.sleep(self.rate_limit_delay)

        return documents

    def _search_term(self, mesh_term: str, retmax: int) -> set[str]:
        """Search PubMed for a single MeSH term.

        Args:
            mesh_term: MeSH term or phrase
            retmax: Max results to return

        Returns:
            Set of PMID strings
        """
        query = (
            f'{mesh_term}[Mesh] AND '
            f'("{self.date_range_start}"[Date - Publication] : "3000"[Date - Publication])'
        )
        try:
            # Entrez.read() parses XML into a dict with top-level capitalized keys
            # (Count, IdList, etc.). Do NOT pass rettype="json" — Entrez.read still
            # expects XML, so json returns get silently discarded as empty lists.
            handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
            result = Entrez.read(handle)
            handle.close()
            pmids = list(result.get("IdList", []))
            logger.info(f"{mesh_term}: {len(pmids)} results")
            return set(pmids)
        except Exception as e:
            logger.error(f"{mesh_term}: ERROR - {e}")
            return set()

    def _fetch_batch(self, pmids: list[str]) -> list[dict[str, Any]]:
        """Fetch full records for a batch of PMIDs.

        Args:
            pmids: List of PMID strings

        Returns:
            List of parsed document dicts
        """
        if not pmids:
            return []

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                retmode="xml",
            )
            records = Entrez.read(handle)
            handle.close()

            articles = records.get("PubmedArticle", [])
            documents = []
            for article in articles:
                doc = self._parse_article(article)
                if doc:
                    documents.append(doc)

            return documents
        except Exception as e:
            logger.error(f"Batch fetch error: {e}")
            return []

    def _parse_article(self, article: dict[str, Any]) -> dict[str, Any] | None:
        """Parse a single PubMed article from XML.

        Args:
            article: Parsed XML article dict from Entrez.read()

        Returns:
            Document dict or None if abstract is missing
        """
        med_data = article.get("MedlineCitation", {})
        article_data = med_data.get("Article", {})
        pubmed_data = article.get("PubmedData", {})

        # Extract PMID
        pmid = med_data.get("PMID", "")
        if not pmid:
            pmid = med_data.get("PMID", {})
            if isinstance(pmid, dict):
                pmid = pmid.get("#text", "")
        pmid = str(pmid)

        # Extract abstract; skip if missing
        abstract_list = article_data.get("Abstract", {})
        if not abstract_list:
            return None

        abstract_texts = abstract_list.get("AbstractText", [])
        if not abstract_texts:
            return None

        # Concatenate all abstract sections
        if isinstance(abstract_texts, str):
            abstract = abstract_texts
        else:
            abstract = " ".join(
                t if isinstance(t, str) else t.get("#text", "")
                for t in (abstract_texts if isinstance(abstract_texts, list) else [abstract_texts])
            )

        if not abstract.strip():
            return None

        # Extract title
        title = article_data.get("ArticleTitle", "")
        if isinstance(title, dict):
            title = title.get("#text", "")
        title = str(title)

        # Extract authors
        author_list = article_data.get("AuthorList", {})
        authors = self._extract_authors(author_list)

        # Extract year
        pub_date_data = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = self._extract_year(pub_date_data)

        # Extract DOI
        article_ids = pubmed_data.get("ArticleIdList", [])
        doi = self._extract_doi(article_ids)

        # Extract MeSH descriptors
        mesh_terms = med_data.get("MeshHeadingList", [])
        mesh_list = self._extract_mesh_terms(mesh_terms)

        # Extract publication types
        pub_types = article_data.get("PublicationTypeList", [])
        pub_type_list = self._extract_publication_types(pub_types)

        return {
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi,
            "mesh_terms": mesh_list,
            "publication_types": pub_type_list,
        }

    def _extract_authors(self, author_list: Any) -> str:
        """Extract author names as a semicolon-separated string.

        biopython returns AuthorList as a ListElement (iterable of Author dicts),
        NOT a DictionaryElement with an "Author" key. Accept either shape for safety.
        """
        if not author_list:
            return ""

        # Handle both ListElement (iterable directly) and old-style {"Author": [...]}
        if hasattr(author_list, "get") and not hasattr(author_list, "__iter__"):
            authors = author_list.get("Author", [])
        elif hasattr(author_list, "get"):
            # DictionaryElement is iterable over keys -- prefer explicit "Author" key
            maybe = author_list.get("Author", None)
            authors = maybe if maybe is not None else list(author_list)
        else:
            authors = list(author_list)

        if not authors:
            return ""

        if not isinstance(authors, list):
            authors = [authors]

        author_strs = []
        for author in authors:
            if not hasattr(author, "get"):
                continue
            last_name = author.get("LastName", "")
            initials = author.get("Initials", "")
            if last_name:
                author_strs.append(f"{last_name} {initials}" if initials else last_name)

        return "; ".join(author_strs)

    def _extract_year(self, pub_date: dict[str, Any]) -> int | None:
        """Extract publication year from PubDate."""
        if not pub_date:
            return None

        # Try Year field first
        year = pub_date.get("Year")
        if year:
            try:
                return int(year)
            except (ValueError, TypeError):
                pass

        # Fall back to MedlineDate
        medline_date = pub_date.get("MedlineDate", "")
        if isinstance(medline_date, dict):
            medline_date = medline_date.get("#text", "")
        if medline_date:
            # Extract first 4-digit number as year
            import re

            match = re.search(r"\b(20\d{2})\b", str(medline_date))
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass

        return None

    def _extract_doi(self, article_ids: list[dict[str, Any]] | dict[str, Any]) -> str:
        """Extract DOI from ArticleIdList."""
        if isinstance(article_ids, dict):
            article_ids = [article_ids]

        for article_id in article_ids:
            if not isinstance(article_id, dict):
                continue
            id_type = article_id.get("IdType", "")
            if id_type == "doi":
                doi_value = article_id.get("#text", "")
                if doi_value:
                    return doi_value

        return ""

    def _extract_mesh_terms(self, mesh_list: list[dict[str, Any]] | dict[str, Any]) -> list[str]:
        """Extract MeSH descriptor names."""
        if not mesh_list:
            return []

        if isinstance(mesh_list, dict):
            mesh_list = [mesh_list]

        terms = []
        for heading in mesh_list:
            if isinstance(heading, dict):
                descriptor = heading.get("DescriptorName", {})
                if isinstance(descriptor, dict):
                    term = descriptor.get("#text", "")
                else:
                    term = str(descriptor)
                if term:
                    terms.append(term)

        return terms

    def _extract_publication_types(self, pub_types: list[dict[str, Any]] | dict[str, Any]) -> list[str]:
        """Extract publication type names."""
        if not pub_types:
            return []

        if isinstance(pub_types, dict):
            pub_types = [pub_types]

        types = []
        for pub_type in pub_types:
            if isinstance(pub_type, dict):
                type_name = pub_type.get("#text", "")
            else:
                type_name = str(pub_type)
            if type_name:
                types.append(type_name)

        return types
