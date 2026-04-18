"""ClinicalTrials.gov document loader (v2 JSON API).

Fetches studies by condition, paginates via nextPageToken, dedupes by NCT ID,
and flattens protocolSection modules into document dicts ready for chunking.

API reference: https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests

from ingest.base import BaseDocumentLoader

logger = logging.getLogger(__name__)


CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"


class ClinicalTrialsLoader(BaseDocumentLoader):
    """Fetch trial summaries from ClinicalTrials.gov v2 API.

    Query strategy: one request per condition string (matched via query.cond
    against conditions/keywords/title). Paginates until max_per_condition is
    reached or no nextPageToken is returned. Dedupes across conditions by NCT
    ID so a trial tagged with multiple conditions is only returned once.
    """

    def __init__(
        self,
        conditions: list[str] | None = None,
        statuses: list[str] | None = None,
        max_per_condition: int = 100,
        start_date_from: int | None = None,
        page_size: int = 50,
        timeout: float = 30.0,
        rate_limit_delay: float = 0.2,
    ):
        """Initialize ClinicalTrials loader.

        Args:
            conditions: CT.gov condition strings (matches conditions/keywords/title)
            statuses: Overall status filter (COMPLETED, RECRUITING, etc.).
                Empty = no filter.
            max_per_condition: Per-condition result cap (caller-side; paginates
                regardless of API default pageSize).
            start_date_from: Minimum study start year. None = no lower bound.
            page_size: API pageSize parameter (v2 max 1000; 50 is a sensible default).
            timeout: Per-request timeout in seconds.
            rate_limit_delay: Seconds between paginated requests. CT.gov v2 is
                public and generous but a small delay keeps us polite.
        """
        self.conditions = conditions or []
        self.statuses = statuses or []
        self.max_per_condition = max_per_condition
        self.start_date_from = start_date_from
        self.page_size = page_size
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._session = requests.Session()

    def validate_config(self) -> bool:
        """Validate loader configuration. Returns True if usable."""
        if not self.conditions:
            raise ValueError("ClinicalTrialsLoader: no conditions configured")
        return True

    def load(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch trials for every configured condition.

        Returns:
            List of flat document dicts (one per unique NCT ID) with keys:
                nct_id, title, summary, detailed_description, conditions,
                keywords, interventions, phases, status, enrollment,
                start_date, completion_date, year, url
        """
        self.validate_config()

        all_docs: dict[str, dict[str, Any]] = {}  # nct_id -> doc
        for condition in self.conditions:
            fetched = self._fetch_condition(condition, self.max_per_condition)
            new_count = 0
            for doc in fetched:
                nct = doc.get("nct_id")
                if nct and nct not in all_docs:
                    all_docs[nct] = doc
                    new_count += 1
            logger.info(f"{condition}: {len(fetched)} fetched, {new_count} new (total unique: {len(all_docs)})")
            time.sleep(self.rate_limit_delay)

        return list(all_docs.values())

    def _fetch_condition(self, condition: str, max_results: int) -> list[dict[str, Any]]:
        """Paginate the v2 studies endpoint for one condition string."""
        docs: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while len(docs) < max_results:
            params: dict[str, Any] = {
                "query.cond": condition,
                "pageSize": min(self.page_size, max_results - len(docs)),
                "format": "json",
            }
            if self.statuses:
                # v2 accepts a comma-separated list of overallStatus values
                params["filter.overallStatus"] = ",".join(self.statuses)
            if next_page_token:
                params["pageToken"] = next_page_token

            try:
                resp = self._session.get(CT_API_BASE, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as e:
                logger.warning(f"{condition}: request error - {type(e).__name__}: {e}")
                return docs

            studies = payload.get("studies", [])
            for study in studies:
                parsed = self._parse_study(study)
                if parsed is None:
                    continue
                # Apply start_date_from filter (v2 has no direct date-from query param,
                # cheapest to filter client-side once parsed).
                if self.start_date_from is not None:
                    year = parsed.get("year")
                    if year is None or year < self.start_date_from:
                        continue
                docs.append(parsed)
                if len(docs) >= max_results:
                    break

            next_page_token = payload.get("nextPageToken")
            if not next_page_token or not studies:
                break
            time.sleep(self.rate_limit_delay)

        return docs

    def _parse_study(self, study: dict[str, Any]) -> dict[str, Any] | None:
        """Flatten a single v2 study record into our document dict shape.

        Returns None if the study lacks both summary and detailed description
        (no retrievable content).
        """
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        desc = proto.get("descriptionModule", {})
        cond_mod = proto.get("conditionsModule", {})
        design = proto.get("designModule", {})
        arms = proto.get("armsInterventionsModule", {})

        nct_id = ident.get("nctId", "")
        if not nct_id:
            return None

        title = ident.get("officialTitle") or ident.get("briefTitle") or ""

        summary = (desc.get("briefSummary") or "").strip()
        detailed = (desc.get("detailedDescription") or "").strip()
        if not summary and not detailed:
            return None

        conditions = cond_mod.get("conditions", []) or []
        keywords = cond_mod.get("keywords", []) or []

        interventions_raw = arms.get("interventions", []) or []
        interventions = [
            f"{i.get('type', '')}: {i.get('name', '')}".strip(": ").strip()
            for i in interventions_raw
            if isinstance(i, dict)
        ]
        interventions = [s for s in interventions if s]

        phases = design.get("phases", []) or []

        start_date = self._date_str(status.get("startDateStruct", {}))
        completion_date = self._date_str(status.get("completionDateStruct", {}))
        year = _year_from_date(start_date)

        enrollment_info = design.get("enrollmentInfo", {}) or {}
        enrollment = enrollment_info.get("count")

        return {
            "nct_id": nct_id,
            "title": title,
            "summary": summary,
            "detailed_description": detailed,
            "conditions": conditions,
            "keywords": keywords,
            "interventions": interventions,
            "phases": phases,
            "status": status.get("overallStatus", ""),
            "enrollment": enrollment,
            "start_date": start_date,
            "completion_date": completion_date,
            "year": year,
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
        }

    @staticmethod
    def _date_str(date_struct: dict[str, Any]) -> str:
        """Pull the date string out of a v2 DateStruct ({date, type})."""
        if not isinstance(date_struct, dict):
            return ""
        return date_struct.get("date", "") or ""


def _year_from_date(s: str) -> int | None:
    """Parse year from CT.gov date strings (YYYY, YYYY-MM, YYYY-MM-DD)."""
    if not s:
        return None
    m = re.match(r"(\d{4})", str(s))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None
