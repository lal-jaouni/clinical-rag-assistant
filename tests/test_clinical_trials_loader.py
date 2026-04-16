"""Tests for the ClinicalTrials.gov v2 loader (Phase 2b).

No network calls: requests.Session.get is monkeypatched to return canned payloads
that mirror the v2 API shape (protocolSection.{identification,status,description,
conditions,design,armsInterventions}Module + nextPageToken pagination).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Match the import style used in scripts/ingest_pubmed.py (src on sys.path).
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingest.clinical_trials_loader import ClinicalTrialsLoader, _year_from_date  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────


def _study(
    nct: str,
    title: str = "Example trial",
    summary: str = "A brief summary of the trial.",
    detailed: str = "",
    conditions: list[str] | None = None,
    keywords: list[str] | None = None,
    interventions: list[dict[str, str]] | None = None,
    phases: list[str] | None = None,
    status: str = "COMPLETED",
    enrollment: int | None = 100,
    start_date: str = "2020-01-15",
    completion_date: str = "2022-06-30",
) -> dict[str, Any]:
    """Build a minimal v2 study record."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct,
                "briefTitle": title,
                "officialTitle": title,
            },
            "statusModule": {
                "overallStatus": status,
                "startDateStruct": {"date": start_date, "type": "ACTUAL"},
                "completionDateStruct": {"date": completion_date, "type": "ACTUAL"},
            },
            "descriptionModule": {
                "briefSummary": summary,
                "detailedDescription": detailed,
            },
            "conditionsModule": {
                "conditions": conditions or ["Hemorrhagic Shock"],
                "keywords": keywords or [],
            },
            "designModule": {
                "phases": phases or ["PHASE2"],
                "enrollmentInfo": {"count": enrollment} if enrollment is not None else {},
            },
            "armsInterventionsModule": {
                "interventions": interventions
                or [{"type": "DRUG", "name": "Tranexamic Acid"}],
            },
        }
    }


class _FakeResponse:
    """Mimic requests.Response for .raise_for_status() and .json()."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


def _install_fake_get(
    monkeypatch: pytest.MonkeyPatch,
    responses_by_condition: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Install a fake Session.get that returns queued payloads per condition.

    Returns the captured-calls list so assertions can inspect request params.
    """
    calls: list[dict[str, Any]] = []

    def _fake_get(self, url, params=None, timeout=None):  # noqa: ARG001
        params = params or {}
        calls.append(dict(params))
        condition = params.get("query.cond", "")
        queue = responses_by_condition.get(condition, [])
        if not queue:
            return _FakeResponse({"studies": []})
        payload = queue.pop(0)
        return _FakeResponse(payload)

    import requests

    monkeypatch.setattr(requests.Session, "get", _fake_get)
    return calls


# ── _year_from_date ──────────────────────────────────────────────────────


class TestYearFromDate:
    def test_full_date(self):
        assert _year_from_date("2020-01-15") == 2020

    def test_year_month(self):
        assert _year_from_date("2021-07") == 2021

    def test_year_only(self):
        assert _year_from_date("2015") == 2015

    def test_empty(self):
        assert _year_from_date("") is None

    def test_junk(self):
        assert _year_from_date("not a date") is None


# ── Config validation ────────────────────────────────────────────────────


class TestValidateConfig:
    def test_raises_when_no_conditions(self):
        loader = ClinicalTrialsLoader(conditions=[])
        with pytest.raises(ValueError, match="no conditions"):
            loader.validate_config()

    def test_passes_with_conditions(self):
        loader = ClinicalTrialsLoader(conditions=["hemorrhagic shock"])
        assert loader.validate_config() is True


# ── _parse_study ─────────────────────────────────────────────────────────


class TestParseStudy:
    def test_extracts_core_fields(self):
        loader = ClinicalTrialsLoader(conditions=["x"])
        doc = loader._parse_study(
            _study(
                "NCT00000001",
                title="TXA in Trauma",
                summary="Study of tranexamic acid.",
                conditions=["Hemorrhage", "Shock"],
                keywords=["TXA", "trauma"],
                phases=["PHASE3"],
                enrollment=500,
                start_date="2018-03-01",
            )
        )
        assert doc is not None
        assert doc["nct_id"] == "NCT00000001"
        assert doc["title"] == "TXA in Trauma"
        assert doc["summary"] == "Study of tranexamic acid."
        assert doc["conditions"] == ["Hemorrhage", "Shock"]
        assert doc["keywords"] == ["TXA", "trauma"]
        assert doc["phases"] == ["PHASE3"]
        assert doc["enrollment"] == 500
        assert doc["start_date"] == "2018-03-01"
        assert doc["year"] == 2018
        assert doc["url"] == "https://clinicaltrials.gov/study/NCT00000001"

    def test_flattens_interventions(self):
        loader = ClinicalTrialsLoader(conditions=["x"])
        doc = loader._parse_study(
            _study(
                "NCT00000002",
                interventions=[
                    {"type": "DRUG", "name": "TXA"},
                    {"type": "PROCEDURE", "name": "Massive Transfusion"},
                ],
            )
        )
        assert doc["interventions"] == ["DRUG: TXA", "PROCEDURE: Massive Transfusion"]

    def test_skips_studies_with_no_description(self):
        loader = ClinicalTrialsLoader(conditions=["x"])
        # Strip both summary and detailed description -> should return None
        study = _study("NCT00000003", summary="", detailed="")
        assert loader._parse_study(study) is None

    def test_skips_studies_with_no_nct_id(self):
        loader = ClinicalTrialsLoader(conditions=["x"])
        study = _study("")
        assert loader._parse_study(study) is None

    def test_falls_back_to_brief_title(self):
        loader = ClinicalTrialsLoader(conditions=["x"])
        study = _study("NCT00000004")
        # Remove officialTitle so briefTitle is used
        study["protocolSection"]["identificationModule"].pop("officialTitle")
        doc = loader._parse_study(study)
        assert doc["title"] == "Example trial"

    def test_handles_missing_enrollment(self):
        loader = ClinicalTrialsLoader(conditions=["x"])
        doc = loader._parse_study(_study("NCT00000005", enrollment=None))
        assert doc["enrollment"] is None


# ── load() end-to-end with fake HTTP ─────────────────────────────────────


class TestLoad:
    def test_single_page(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_get(
            monkeypatch,
            {
                "hemorrhagic shock": [
                    {
                        "studies": [_study("NCT0001"), _study("NCT0002")],
                        "nextPageToken": None,
                    }
                ]
            },
        )
        loader = ClinicalTrialsLoader(
            conditions=["hemorrhagic shock"],
            max_per_condition=50,
            rate_limit_delay=0,
        )
        docs = loader.load()
        ids = sorted(d["nct_id"] for d in docs)
        assert ids == ["NCT0001", "NCT0002"]

    def test_pagination(self, monkeypatch: pytest.MonkeyPatch):
        # Two pages; loader must follow nextPageToken.
        _install_fake_get(
            monkeypatch,
            {
                "trauma": [
                    {
                        "studies": [_study("NCT_A"), _study("NCT_B")],
                        "nextPageToken": "tok-2",
                    },
                    {
                        "studies": [_study("NCT_C")],
                        "nextPageToken": None,
                    },
                ]
            },
        )
        loader = ClinicalTrialsLoader(
            conditions=["trauma"],
            max_per_condition=50,
            page_size=2,
            rate_limit_delay=0,
        )
        docs = loader.load()
        assert sorted(d["nct_id"] for d in docs) == ["NCT_A", "NCT_B", "NCT_C"]

    def test_dedup_across_conditions(self, monkeypatch: pytest.MonkeyPatch):
        # Same NCT returned under two conditions -> single entry in output.
        _install_fake_get(
            monkeypatch,
            {
                "cond1": [{"studies": [_study("NCT_DUP"), _study("NCT_X")]}],
                "cond2": [{"studies": [_study("NCT_DUP"), _study("NCT_Y")]}],
            },
        )
        loader = ClinicalTrialsLoader(
            conditions=["cond1", "cond2"],
            max_per_condition=50,
            rate_limit_delay=0,
        )
        docs = loader.load()
        assert sorted(d["nct_id"] for d in docs) == ["NCT_DUP", "NCT_X", "NCT_Y"]

    def test_max_per_condition_caps_results(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_get(
            monkeypatch,
            {
                "cond": [
                    {
                        "studies": [_study(f"NCT_{i}") for i in range(10)],
                        "nextPageToken": None,
                    }
                ]
            },
        )
        loader = ClinicalTrialsLoader(
            conditions=["cond"],
            max_per_condition=3,
            rate_limit_delay=0,
        )
        docs = loader.load()
        assert len(docs) == 3

    def test_start_date_filter(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_get(
            monkeypatch,
            {
                "cond": [
                    {
                        "studies": [
                            _study("NCT_OLD", start_date="2005-01-01"),
                            _study("NCT_NEW", start_date="2020-03-10"),
                            _study("NCT_MID", start_date="2012-06-01"),
                        ],
                        "nextPageToken": None,
                    }
                ]
            },
        )
        loader = ClinicalTrialsLoader(
            conditions=["cond"],
            max_per_condition=50,
            start_date_from=2010,
            rate_limit_delay=0,
        )
        docs = loader.load()
        ids = sorted(d["nct_id"] for d in docs)
        assert ids == ["NCT_MID", "NCT_NEW"]

    def test_status_filter_passed_to_api(self, monkeypatch: pytest.MonkeyPatch):
        calls = _install_fake_get(
            monkeypatch,
            {"cond": [{"studies": [_study("NCT1")], "nextPageToken": None}]},
        )
        loader = ClinicalTrialsLoader(
            conditions=["cond"],
            statuses=["COMPLETED", "RECRUITING"],
            max_per_condition=10,
            rate_limit_delay=0,
        )
        loader.load()
        assert calls[0]["filter.overallStatus"] == "COMPLETED,RECRUITING"

    def test_no_status_filter_when_empty(self, monkeypatch: pytest.MonkeyPatch):
        calls = _install_fake_get(
            monkeypatch,
            {"cond": [{"studies": [_study("NCT1")], "nextPageToken": None}]},
        )
        loader = ClinicalTrialsLoader(
            conditions=["cond"],
            statuses=[],
            max_per_condition=10,
            rate_limit_delay=0,
        )
        loader.load()
        assert "filter.overallStatus" not in calls[0]

    def test_http_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        """A network/HTTP failure on a condition should not crash the whole load."""

        def _boom(self, url, params=None, timeout=None):  # noqa: ARG001
            raise RuntimeError("network down")

        import requests

        monkeypatch.setattr(requests.Session, "get", _boom)
        loader = ClinicalTrialsLoader(
            conditions=["cond_a", "cond_b"],
            max_per_condition=10,
            rate_limit_delay=0,
        )
        docs = loader.load()
        assert docs == []
