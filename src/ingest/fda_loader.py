"""FDA SaMD / AI-ML guidance document loader.

Downloads FDA guidance PDFs (GMLP, PCCP, CDS, SaMD clinical evaluation, etc.),
caches them locally, extracts text via pdfplumber, and heuristically parses
Roman-numeral section headers (I. INTRODUCTION, II. BACKGROUND, ...).

Return shape mirrors the other loaders so the same chunker + ingestion runner
can consume it:

    {
        "fda_doc_id": str,       # stable identifier (e.g., "gmlp-2021")
        "title": str,
        "doc_type": str,         # "Guidance", "Action Plan", ...
        "issuance_date": str,    # "YYYY-MM-DD" or "" if unknown
        "year": int | None,
        "url": str,
        "text": str,             # full extracted body
        "sections": list[dict],  # [{"name": "...", "text": "..."}, ...]
        "pdf_path": str,         # local cache path (for provenance)
    }
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import requests

from ingest.base import BaseDocumentLoader

logger = logging.getLogger(__name__)


# Matches section headers like:
#   "I. INTRODUCTION"
#   "II. BACKGROUND"
#   "IX. DEVICE SOFTWARE FUNCTIONS"
# Requires uppercase body text to avoid catching "II. items in a list"
# or mid-paragraph references.
_ROMAN_HEADER_RE = re.compile(
    r"^\s*(?P<num>[IVX]{1,5})\.\s+(?P<name>[A-Z][A-Z0-9 ,\-&/()]{2,80})\s*$",
    re.MULTILINE,
)


class FDALoader(BaseDocumentLoader):
    """Fetch and parse FDA AI/ML SaMD guidance PDFs.

    The document list is config-driven: each entry names an ID, title, source
    URL, and optional metadata. The loader caches downloads under cache_dir,
    so re-running the pipeline doesn't re-hit FDA's servers.
    """

    def __init__(
        self,
        documents: list[dict[str, Any]] | None = None,
        cache_dir: str = "/tmp/fda_cache",
        timeout: float = 60.0,
    ):
        """Initialize the FDA loader.

        Args:
            documents: List of doc manifests. Each entry should provide:
                doc_id (str, required)  -- stable ID used as source_id
                title (str, required)
                url (str, required)     -- direct PDF URL
                doc_type (str, optional)
                issuance_date (str, optional)  -- "YYYY-MM-DD"
                local_path (str, optional)     -- pre-downloaded PDF; skips
                    network fetch entirely.
            cache_dir: Where downloaded PDFs live between runs.
            timeout: Per-request HTTP timeout.
        """
        self.documents = documents or []
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self._session = requests.Session()
        # fda.gov serves a bot-detection apology page to clients that
        # identify as python-requests. Pretend to be a normal desktop browser
        # so the CDN hands us the real PDF binary.
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "application/pdf,*/*;q=0.8",
            }
        )

    # ── Public API ───────────────────────────────────────────────────────

    def validate_config(self) -> bool:
        """Validate loader configuration."""
        if not self.documents:
            raise ValueError("FDALoader: no documents configured")
        missing = [
            d.get("doc_id", "<?>")
            for d in self.documents
            if not (d.get("doc_id") and d.get("title") and (d.get("url") or d.get("local_path")))
        ]
        if missing:
            raise ValueError(
                f"FDALoader: entries missing required fields (doc_id/title/url): {missing}"
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return True

    def load(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Download (or re-use cached) PDFs and extract text + sections.

        Fails soft: a single PDF that 404s or fails to parse is skipped with a
        printed warning rather than aborting the whole batch.
        """
        self.validate_config()

        results: list[dict[str, Any]] = []
        for entry in self.documents:
            doc_id = entry["doc_id"]
            try:
                pdf_path = self._ensure_pdf(entry)
            except Exception as e:
                logger.warning(f"[skip] {doc_id}: download failed: {type(e).__name__}: {e}")
                continue

            try:
                text = self._extract_text(pdf_path)
            except Exception as e:
                logger.warning(f"[skip] {doc_id}: text extraction failed: {type(e).__name__}: {e}")
                continue

            if not text.strip():
                logger.warning(f"[skip] {doc_id}: extracted text is empty")
                continue

            sections = self._parse_sections(text)
            issuance_date = entry.get("issuance_date", "") or ""
            year = _year_from_date(issuance_date)

            results.append(
                {
                    "fda_doc_id": doc_id,
                    "title": entry["title"],
                    "doc_type": entry.get("doc_type", "Guidance"),
                    "issuance_date": issuance_date,
                    "year": year,
                    "url": entry.get("url", ""),
                    "text": text,
                    "sections": sections,
                    "pdf_path": str(pdf_path),
                }
            )
            logger.info(f"[ok] {doc_id}: {len(text)} chars, {len(sections)} sections")

        return results

    # ── Internals ────────────────────────────────────────────────────────

    def _ensure_pdf(self, entry: dict[str, Any]) -> Path:
        """Resolve a doc entry to a local PDF path, downloading if needed."""
        local = entry.get("local_path")
        if local:
            p = Path(local)
            if not p.exists():
                raise FileNotFoundError(f"local_path does not exist: {local}")
            return p

        cache_path = self.cache_dir / f"{entry['doc_id']}.pdf"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path

        url = entry["url"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        resp = self._session.get(url, timeout=self.timeout, stream=True, allow_redirects=True)
        resp.raise_for_status()
        content = resp.content
        if not content:
            raise RuntimeError("empty response body")
        # Guard against fda.gov's bot-apology page, which returns HTTP 200 with
        # an HTML body rather than the PDF we asked for.
        if not content.startswith(b"%PDF"):
            ctype = resp.headers.get("Content-Type", "unknown")
            raise RuntimeError(
                f"response is not a PDF (Content-Type={ctype}, starts with "
                f"{content[:16]!r}, final URL={resp.url})"
            )
        cache_path.write_bytes(content)
        return cache_path

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract full-document text by concatenating per-page text."""
        import pdfplumber  # local import keeps top-of-module light

        pages: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)
        return "\n\n".join(pages)

    def _parse_sections(self, text: str) -> list[dict[str, str]]:
        """Heuristically split body text into sections keyed by header lines.

        Uses Roman-numeral headers (I., II., ...) which FDA guidance
        documents use consistently. When no headers are found we return one
        synthetic "BODY" section so downstream code can rely on the key.
        """
        matches = list(_ROMAN_HEADER_RE.finditer(text))
        if not matches:
            return [{"name": "BODY", "text": text.strip()}]

        sections: list[dict[str, str]] = []
        # Preamble before the first header (title page, TOC, etc.)
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append({"name": "PREAMBLE", "text": preamble})

        for i, m in enumerate(matches):
            name = f"{m.group('num')}. {m.group('name').strip()}"
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            if body:
                sections.append({"name": name, "text": body})

        return sections


def _year_from_date(date_str: str) -> int | None:
    """Parse leading year out of a YYYY-MM-DD (or similar) date string."""
    if not date_str:
        return None
    m = re.match(r"(\d{4})", date_str)
    return int(m.group(1)) if m else None
