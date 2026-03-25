"""
Unit tests for src/tools/web_search.py — WebSearchTool.

DuckDuckGo is fully mocked — no network access required.

IMPORTANT: DDGS is imported lazily *inside* the tool function, so we patch
'duckduckgo_search.DDGS' (the canonical import location) rather than an
attribute on the src module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools.web_search import web_search

pytestmark = pytest.mark.unit

# Sample DDGS result dictionaries
_FAKE_RESULTS = [
    {
        "title": "Perovskite Solar Cells Reach 29% Efficiency",
        "body": "Researchers report record efficiency for perovskite-silicon tandem cells.",
        "href": "https://example.com/perovskite-solar",
    },
    {
        "title": "NIST Releases New Cybersecurity Framework 2.0",
        "body": "The updated framework adds governance as a sixth pillar.",
        "href": "https://example.com/nist-csf",
    },
    {
        "title": "BaTiO3 XRD Characterization Review",
        "body": "Comprehensive review of XRD peaks for barium titanate polymorphs.",
        "href": "https://example.com/batio3-xrd",
    },
]


# ---------------------------------------------------------------------------
# Helper: build a DDGS mock that behaves as a context manager
# ---------------------------------------------------------------------------


def _make_ddgs_mock(results: list) -> MagicMock:
    """Return a mock DDGS class whose instances act as context managers."""
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)
    mock_instance.text.return_value = results

    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls, mock_instance


# ---------------------------------------------------------------------------
# Successful search
# ---------------------------------------------------------------------------


class TestWebSearchSuccess:
    def test_returns_string(self):
        mock_cls, _ = _make_ddgs_mock(_FAKE_RESULTS)
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("perovskite XRD")
        assert isinstance(result, str)

    def test_contains_all_titles(self):
        mock_cls, _ = _make_ddgs_mock(_FAKE_RESULTS)
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("perovskite")
        assert "Perovskite Solar Cells" in result
        assert "NIST" in result
        assert "BaTiO3" in result

    def test_contains_urls(self):
        mock_cls, _ = _make_ddgs_mock(_FAKE_RESULTS)
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("test query")
        assert "https://example.com" in result

    def test_numbered_results(self):
        """Results should be numbered [1], [2], [3]."""
        mock_cls, _ = _make_ddgs_mock(_FAKE_RESULTS)
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("anything")
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_query_echoed_in_output(self):
        mock_cls, _ = _make_ddgs_mock(_FAKE_RESULTS)
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("my special query")
        assert "my special query" in result


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


class TestWebSearchEmptyResults:
    def test_empty_results_returns_no_results_message(self):
        mock_cls, _ = _make_ddgs_mock([])
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("obscure query with no results")
        assert "no results" in result.lower()

    def test_empty_results_contains_query(self):
        mock_cls, _ = _make_ddgs_mock([])
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("xyzzy_nonexistent")
        assert "xyzzy_nonexistent" in result


# ---------------------------------------------------------------------------
# Network / timeout errors
# ---------------------------------------------------------------------------


class TestWebSearchErrors:
    def test_network_timeout_returns_error_string(self):
        mock_cls, mock_instance = _make_ddgs_mock([])
        mock_instance.text.side_effect = Exception("Connection timed out")
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("timeout test")
        assert "failed" in result.lower() or "error" in result.lower()
        assert isinstance(result, str)

    def test_connection_error_no_exception_raised(self):
        """Tool must never propagate exceptions to the caller."""
        mock_cls, mock_instance = _make_ddgs_mock([])
        mock_instance.text.side_effect = ConnectionError("Network unreachable")
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("no network")
        assert isinstance(result, str)

    def test_import_error_returns_install_message(self):
        """If duckduckgo_search is not installed, return install instruction."""
        import importlib
        import sys

        # Remove the module from sys.modules to simulate non-installation,
        # then patch the import so ImportError is raised inside the function.
        original = sys.modules.get("duckduckgo_search")
        try:
            sys.modules["duckduckgo_search"] = None  # type: ignore[assignment]
            result = web_search.invoke("test import error")
        finally:
            if original is None:
                sys.modules.pop("duckduckgo_search", None)
            else:
                sys.modules["duckduckgo_search"] = original

        assert isinstance(result, str)
        assert "pip install" in result.lower() or "not installed" in result.lower()


# ---------------------------------------------------------------------------
# Result format edge cases
# ---------------------------------------------------------------------------


class TestWebSearchResultFormat:
    def test_missing_body_falls_back_to_snippet(self):
        """Result with 'snippet' key instead of 'body' still works."""
        results = [
            {
                "title": "Test Title",
                "snippet": "A snippet here",
                "href": "https://example.com",
            }
        ]
        mock_cls, _ = _make_ddgs_mock(results)
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("test")
        assert "Test Title" in result

    def test_missing_href_falls_back_to_url(self):
        """Result with 'url' key instead of 'href' still works."""
        results = [
            {
                "title": "Fallback Test",
                "body": "Body text",
                "url": "https://fallback.com",
            }
        ]
        mock_cls, _ = _make_ddgs_mock(results)
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("test")
        assert "Fallback Test" in result

    def test_single_result(self):
        results = [_FAKE_RESULTS[0]]
        mock_cls, _ = _make_ddgs_mock(results)
        with patch("duckduckgo_search.DDGS", mock_cls):
            result = web_search.invoke("single")
        assert "[1]" in result
        assert "[2]" not in result
