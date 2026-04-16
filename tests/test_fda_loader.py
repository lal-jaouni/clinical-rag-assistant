"""Tests for the FDA SaMD loader (Phase 2c).

No network calls: requests.Session.get and pdfplumber.open are monkeypatched
so tests run offline and deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingest.fda_loader import FDALoader, _year_from_date  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────


SAMPLE_GUIDANCE_TEXT = """\
Guidance for Industry and FDA Staff

Software as a Medical Device

Document issued on October 27, 2021.

I. INTRODUCTION

This guidance document describes FDA's current thinking on good machine
learning practice for medical device development. It is non-binding.

II. BACKGROUND

AI/ML-enabled devices are increasingly common. This section outlines
the regulatory landscape and motivates the guiding principles below.

III. GUIDING PRINCIPLES

The following ten principles apply across the total product lifecycle:
1. Multi-disciplinary expertise is leveraged throughout the lifecycle.
2. Good software engineering and security practices are implemented.

IV. CONCLUSION

These principles form a foundation for further industry engagement.
"""


def _make_fake_pdfplumber(monkeypatch: pytest.MonkeyPatch, pages_by_path: dict[str, list[str]]) -> None:
    """Install a fake pdfplumber.open that returns canned page text per path."""

    class _FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakePDF:
        def __init__(self, pages: list[str]):
            self.pages = [_FakePage(t) for t in pages]

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    def _fake_open(path: str) -> _FakePDF:
        # Normalize to absolute string path for matching convenience
        key = str(path)
        pages = pages_by_path.get(key, pages_by_path.get(Path(key).name, []))
        return _FakePDF(pages)

    import pdfplumber

    monkeypatch.setattr(pdfplumber, "open", _fake_open)


def _install_fake_session_get(monkeypatch: pytest.MonkeyPatch, responses: dict[str, bytes]) -> list[str]:
    """Install a fake Session.get returning PDF bytes by URL. Returns call log."""
    calls: list[str] = []

    class _FakeResponse:
        def __init__(self, content: bytes, status: int = 200):
            self.content = content
            self.status_code = status

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    def _fake_get(self, url, timeout=None, stream=False):  # noqa: ARG001
        calls.append(url)
        if url not in responses:
            return _FakeResponse(b"", status=404)
        return _FakeResponse(responses[url])

    import requests

    monkeypatch.setattr(requests.Session, "get", _fake_get)
    return calls


# ── _year_from_date ──────────────────────────────────────────────────────


class TestYearFromDate:
    def test_full_date(self):
        assert _year_from_date("2021-10-27") == 2021

    def test_year_only(self):
        assert _year_from_date("2017") == 2017

    def test_empty(self):
        assert _year_from_date("") is None

    def test_junk(self):
        assert _year_from_date("not a date") is None


# ── Config validation ────────────────────────────────────────────────────


class TestValidateConfig:
    def test_raises_when_no_documents(self, tmp_path):
        loader = FDALoader(documents=[], cache_dir=str(tmp_path))
        with pytest.raises(ValueError, match="no documents"):
            loader.validate_config()

    def test_raises_when_entry_missing_fields(self, tmp_path):
        loader = FDALoader(
            documents=[{"doc_id": "x", "title": "X"}],  # no url / local_path
            cache_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match="missing required fields"):
            loader.validate_config()

    def test_passes_with_url(self, tmp_path):
        loader = FDALoader(
            documents=[{"doc_id": "x", "title": "X", "url": "http://example/x.pdf"}],
            cache_dir=str(tmp_path / "cache"),
        )
        assert loader.validate_config() is True
        # Cache dir is created eagerly
        assert (tmp_path / "cache").exists()

    def test_passes_with_local_path(self, tmp_path):
        pdf = tmp_path / "local.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        loader = FDALoader(
            documents=[{"doc_id": "x", "title": "X", "local_path": str(pdf)}],
            cache_dir=str(tmp_path / "cache"),
        )
        assert loader.validate_config() is True


# ── _parse_sections ──────────────────────────────────────────────────────


class TestParseSections:
    def test_roman_headers(self, tmp_path):
        loader = FDALoader(cache_dir=str(tmp_path))
        sections = loader._parse_sections(SAMPLE_GUIDANCE_TEXT)

        names = [s["name"] for s in sections]
        # Expect PREAMBLE then four Roman-numeral sections
        assert names[0] == "PREAMBLE"
        assert "I. INTRODUCTION" in names
        assert "II. BACKGROUND" in names
        assert "III. GUIDING PRINCIPLES" in names
        assert "IV. CONCLUSION" in names

        intro = next(s for s in sections if s["name"] == "I. INTRODUCTION")
        # Whitespace-normalize before substring check so the wrapped
        # "good machine\nlearning" doesn't trip us up.
        assert "good machine learning practice" in " ".join(intro["text"].split())
        # Body text should NOT contain the next header line
        assert "II. BACKGROUND" not in intro["text"]

    def test_no_headers_returns_single_body(self, tmp_path):
        loader = FDALoader(cache_dir=str(tmp_path))
        sections = loader._parse_sections("Just some flat guidance body text with no headers.")
        assert len(sections) == 1
        assert sections[0]["name"] == "BODY"
        assert "guidance body" in sections[0]["text"]

    def test_empty_text_returns_single_body(self, tmp_path):
        loader = FDALoader(cache_dir=str(tmp_path))
        sections = loader._parse_sections("")
        assert sections == [{"name": "BODY", "text": ""}]

    def test_lowercase_body_not_matched_as_header(self, tmp_path):
        """'I. lowercase body' should NOT be treated as a section header."""
        loader = FDALoader(cache_dir=str(tmp_path))
        text = "Some preamble.\n\nI. lowercase body\n\nmore content."
        sections = loader._parse_sections(text)
        assert len(sections) == 1
        assert sections[0]["name"] == "BODY"


# ── _ensure_pdf ──────────────────────────────────────────────────────────


class TestEnsurePdf:
    def test_uses_local_path_when_provided(self, tmp_path):
        pdf = tmp_path / "local.pdf"
        pdf.write_bytes(b"%PDF-1.4\nlocal")
        loader = FDALoader(cache_dir=str(tmp_path / "cache"))
        path = loader._ensure_pdf({"doc_id": "x", "local_path": str(pdf)})
        assert path == pdf

    def test_raises_when_local_path_missing(self, tmp_path):
        loader = FDALoader(cache_dir=str(tmp_path / "cache"))
        with pytest.raises(FileNotFoundError):
            loader._ensure_pdf({"doc_id": "x", "local_path": str(tmp_path / "nope.pdf")})

    def test_uses_cache_when_present(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "x.pdf").write_bytes(b"%PDF-1.4\ncached")
        loader = FDALoader(cache_dir=str(cache))
        path = loader._ensure_pdf({"doc_id": "x", "url": "http://example/x.pdf"})
        assert path == cache / "x.pdf"
        assert path.read_bytes() == b"%PDF-1.4\ncached"

    def test_downloads_and_caches(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        loader = FDALoader(cache_dir=str(cache))
        calls = _install_fake_session_get(
            monkeypatch, {"http://example/x.pdf": b"%PDF-1.4\ndownloaded"}
        )
        path = loader._ensure_pdf({"doc_id": "x", "url": "http://example/x.pdf"})
        assert path == cache / "x.pdf"
        assert path.read_bytes() == b"%PDF-1.4\ndownloaded"
        assert calls == ["http://example/x.pdf"]

    def test_download_failure_raises(self, tmp_path, monkeypatch):
        loader = FDALoader(cache_dir=str(tmp_path / "cache"))
        _install_fake_session_get(monkeypatch, {})  # 404 for any URL
        with pytest.raises(RuntimeError, match="HTTP 404"):
            loader._ensure_pdf({"doc_id": "x", "url": "http://example/missing.pdf"})


# ── _extract_text ────────────────────────────────────────────────────────


class TestExtractText:
    def test_concatenates_page_text(self, tmp_path, monkeypatch):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        _make_fake_pdfplumber(monkeypatch, {str(pdf): ["Page one.", "Page two."]})
        loader = FDALoader(cache_dir=str(tmp_path))
        text = loader._extract_text(pdf)
        assert "Page one." in text
        assert "Page two." in text

    def test_skips_empty_pages(self, tmp_path, monkeypatch):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        _make_fake_pdfplumber(monkeypatch, {str(pdf): ["", "Content.", "   "]})
        loader = FDALoader(cache_dir=str(tmp_path))
        text = loader._extract_text(pdf)
        assert text.strip() == "Content."


# ── load() end-to-end (offline) ──────────────────────────────────────────


class TestLoad:
    def _build_loader(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        pdf_bytes = b"%PDF-1.4\nbytes-dont-actually-matter-because-we-stub-pdfplumber"

        calls = _install_fake_session_get(
            monkeypatch,
            {
                "http://example/gmlp.pdf": pdf_bytes,
                "http://example/cds.pdf": pdf_bytes,
            },
        )
        # After _ensure_pdf writes cache, the pdfplumber stub will be keyed on
        # the cache path's filename.
        _make_fake_pdfplumber(
            monkeypatch,
            {
                "gmlp.pdf": [SAMPLE_GUIDANCE_TEXT],
                "cds.pdf": ["Flat body text without section headers at all."],
            },
        )

        loader = FDALoader(
            documents=[
                {
                    "doc_id": "gmlp",
                    "title": "GMLP",
                    "doc_type": "Guiding Principles",
                    "issuance_date": "2021-10-27",
                    "url": "http://example/gmlp.pdf",
                },
                {
                    "doc_id": "cds",
                    "title": "CDS",
                    "doc_type": "Final Guidance",
                    "issuance_date": "2022-09-28",
                    "url": "http://example/cds.pdf",
                },
            ],
            cache_dir=str(cache),
        )
        return loader, calls

    def test_returns_structured_documents(self, tmp_path, monkeypatch):
        loader, _calls = self._build_loader(tmp_path, monkeypatch)
        docs = loader.load()
        assert len(docs) == 2

        by_id = {d["fda_doc_id"]: d for d in docs}
        gmlp = by_id["gmlp"]
        assert gmlp["title"] == "GMLP"
        assert gmlp["doc_type"] == "Guiding Principles"
        assert gmlp["year"] == 2021
        assert gmlp["issuance_date"] == "2021-10-27"
        assert "I. INTRODUCTION" in [s["name"] for s in gmlp["sections"]]
        assert gmlp["pdf_path"].endswith("gmlp.pdf")
        assert "good machine" in gmlp["text"]

        cds = by_id["cds"]
        assert cds["year"] == 2022
        # No headers in cds.pdf -> single BODY section
        assert len(cds["sections"]) == 1
        assert cds["sections"][0]["name"] == "BODY"

    def test_skips_failed_download(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        _install_fake_session_get(monkeypatch, {})  # every URL 404s
        _make_fake_pdfplumber(monkeypatch, {})

        loader = FDALoader(
            documents=[
                {"doc_id": "x", "title": "X", "url": "http://example/missing.pdf"},
            ],
            cache_dir=str(cache),
        )
        docs = loader.load()
        assert docs == []

    def test_skips_empty_text(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        _install_fake_session_get(monkeypatch, {"http://example/x.pdf": b"%PDF-1.4\n"})
        _make_fake_pdfplumber(monkeypatch, {"x.pdf": ["", "   "]})

        loader = FDALoader(
            documents=[{"doc_id": "x", "title": "X", "url": "http://example/x.pdf"}],
            cache_dir=str(cache),
        )
        docs = loader.load()
        assert docs == []

    def test_reuses_cache_on_second_call(self, tmp_path, monkeypatch):
        loader, calls = self._build_loader(tmp_path, monkeypatch)
        loader.load()
        first_call_count = len(calls)
        loader.load()
        # Second invocation should not re-hit the network because PDFs are cached
        assert len(calls) == first_call_count
