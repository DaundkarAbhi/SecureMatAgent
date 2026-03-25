"""
Integration tests for src/api/main.py — FastAPI endpoints.

Uses FastAPI TestClient (synchronous wrapper over ASGI).
All external dependencies (agent, ingestion pipeline, Ollama, Qdrant) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# App fixture — import app once with all heavy deps mocked
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """
    Build TestClient with agent and ingestion pipeline mocked at the
    module level so FastAPI startup doesn't attempt real connections.
    """
    # Patch ask() so /api/chat doesn't need real LLM
    mock_ask = MagicMock(
        return_value={
            "answer": "The 2θ angle for d=2.87 Å is approximately 31.1 degrees.",
            "tools_used": ["materials_calculator"],
            "sources": ["perovskite_XRD.pdf"],
            "latency_ms": 42.0,
        }
    )

    # Patch run_ingestion so /api/ingest doesn't need real pipeline
    mock_ingest = MagicMock(
        return_value={
            "total_files": 3,
            "total_chunks": 15,
            "collection_size": 15,
            "domain_breakdown": {"materials_science": 10, "cybersecurity": 5},
        }
    )

    with (
        patch("src.api.main.ask", mock_ask, create=True),
        patch("src.ingestion.pipeline.run_ingestion", mock_ingest),
    ):
        from src.api.main import app

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    def test_chat_returns_200(self, client):
        resp = client.post(
            "/api/chat",
            json={"query": "What is the Bragg angle for d=2.87 Å?"},
        )
        assert resp.status_code == 200

    def test_chat_response_schema(self, client):
        resp = client.post("/api/chat", json={"query": "test query"})
        data = resp.json()
        assert "answer" in data
        assert "tools_used" in data
        assert "sources" in data
        assert "latency_ms" in data
        assert "session_id" in data

    def test_chat_answer_is_string(self, client):
        resp = client.post("/api/chat", json={"query": "hello"})
        data = resp.json()
        assert isinstance(data["answer"], str)

    def test_chat_tools_used_is_list(self, client):
        resp = client.post("/api/chat", json={"query": "hello"})
        data = resp.json()
        assert isinstance(data["tools_used"], list)

    def test_chat_sources_is_list(self, client):
        resp = client.post("/api/chat", json={"query": "hello"})
        data = resp.json()
        assert isinstance(data["sources"], list)

    def test_chat_with_session_id(self, client):
        resp = client.post(
            "/api/chat",
            json={"query": "test", "session_id": "my-session-99"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "my-session-99"

    def test_chat_empty_query_returns_422(self, client):
        """Empty string fails Pydantic min_length=1 validation → 422."""
        resp = client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 422

    def test_chat_missing_query_returns_422(self, client):
        """Missing required field → 422 Unprocessable Entity."""
        resp = client.post("/api/chat", json={"session_id": "s1"})
        assert resp.status_code == 422

    def test_chat_null_query_returns_422(self, client):
        resp = client.post("/api/chat", json={"query": None})
        assert resp.status_code == 422

    def test_chat_agent_error_returns_500(self):
        """When ask() returns an 'Agent error:' answer → 500."""
        error_result = {
            "answer": "Agent error: LLM unreachable",
            "tools_used": [],
            "sources": [],
            "latency_ms": 1.0,
        }
        # ask() is imported inside _run_agent() → patch at source module
        with patch("src.agent.run.ask", MagicMock(return_value=error_result)):
            from src.api.main import app

            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/api/chat", json={"query": "trigger error"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200_when_services_mocked_ok(self):
        """Mock both health checks to return 200 → overall status ok."""
        from src.api.models import HealthStatus

        ok_status = HealthStatus(status="ok", message="Reachable")

        with (
            patch("src.api.main._check_ollama", new=AsyncMock(return_value=ok_status)),
            patch("src.api.main._check_qdrant", new=AsyncMock(return_value=ok_status)),
        ):
            from src.api.main import app

            with TestClient(app) as c:
                resp = c.get("/api/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_response_has_ollama_and_qdrant(self):
        from src.api.models import HealthStatus

        ok = HealthStatus(status="ok", message="up")
        with (
            patch("src.api.main._check_ollama", new=AsyncMock(return_value=ok)),
            patch("src.api.main._check_qdrant", new=AsyncMock(return_value=ok)),
        ):
            from src.api.main import app

            with TestClient(app) as c:
                resp = c.get("/api/health")

        data = resp.json()
        assert "ollama" in data
        assert "qdrant" in data

    def test_health_degraded_when_ollama_error(self):
        from src.api.models import HealthStatus

        ok = HealthStatus(status="ok", message="up")
        err = HealthStatus(status="error", message="Connection refused")

        with (
            patch("src.api.main._check_ollama", new=AsyncMock(return_value=err)),
            patch("src.api.main._check_qdrant", new=AsyncMock(return_value=ok)),
        ):
            from src.api.main import app

            with TestClient(app) as c:
                resp = c.get("/api/health")

        data = resp.json()
        assert data["status"] in ("degraded", "error")

    def test_health_error_when_both_services_down(self):
        from src.api.models import HealthStatus

        err = HealthStatus(status="error", message="down")
        with (
            patch("src.api.main._check_ollama", new=AsyncMock(return_value=err)),
            patch("src.api.main._check_qdrant", new=AsyncMock(return_value=err)),
        ):
            from src.api.main import app

            with TestClient(app) as c:
                resp = c.get("/api/health")

        data = resp.json()
        assert data["status"] == "error"


# ---------------------------------------------------------------------------
# POST /api/ingest
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    def test_ingest_returns_200(self, tmp_path):
        mock_result = {
            "total_files": 2,
            "total_chunks": 10,
            "collection_size": 10,
            "domain_breakdown": {"materials_science": 10},
        }
        with patch("src.ingestion.pipeline.run_ingestion", return_value=mock_result):
            from src.api.main import app

            with TestClient(app) as c:
                resp = c.post("/api/ingest", json={"data_dir": str(tmp_path)})

        assert resp.status_code == 200

    def test_ingest_response_schema(self, tmp_path):
        mock_result = {
            "total_files": 1,
            "total_chunks": 5,
            "collection_size": 5,
            "domain_breakdown": {"general": 5},
        }
        with patch("src.ingestion.pipeline.run_ingestion", return_value=mock_result):
            from src.api.main import app

            with TestClient(app) as c:
                resp = c.post("/api/ingest", json={"data_dir": str(tmp_path)})

        data = resp.json()
        assert "status" in data
        assert "total_files" in data
        assert "total_chunks" in data
        assert "domain_breakdown" in data

    def test_ingest_missing_data_dir_returns_422(self):
        from src.api.main import app

        with TestClient(app) as c:
            resp = c.post("/api/ingest", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}/history
# ---------------------------------------------------------------------------


class TestSessionHistoryEndpoint:
    def test_history_returns_200_for_new_session(self):
        from src.api.main import app

        with TestClient(app) as c:
            resp = c.get("/api/sessions/brand-new-session/history")
        assert resp.status_code == 200

    def test_history_response_schema(self):
        from src.api.main import app

        with TestClient(app) as c:
            resp = c.get("/api/sessions/test-session-history/history")
        data = resp.json()
        assert "session_id" in data
        assert "messages" in data
        assert "message_count" in data

    def test_history_empty_for_unknown_session(self):
        from src.api.main import app

        with TestClient(app) as c:
            resp = c.get("/api/sessions/totally-unknown-xyz/history")
        data = resp.json()
        assert data["message_count"] == 0
        assert data["messages"] == []


# ---------------------------------------------------------------------------
# DELETE /api/sessions/{session_id}
# ---------------------------------------------------------------------------


class TestDeleteSessionEndpoint:
    def test_delete_returns_200(self):
        from src.api.main import app

        with TestClient(app) as c:
            resp = c.delete("/api/sessions/delete-me-session")
        assert resp.status_code == 200

    def test_delete_response_schema(self):
        from src.api.main import app

        with TestClient(app) as c:
            resp = c.delete("/api/sessions/some-session-del")
        data = resp.json()
        assert "session_id" in data
        assert "cleared" in data
        assert "message" in data

    def test_delete_nonexistent_session(self):
        """Deleting a session that never existed should return 200 with cleared=False."""
        from src.api.main import app

        with TestClient(app) as c:
            resp = c.delete("/api/sessions/never-existed-abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cleared"] is False
