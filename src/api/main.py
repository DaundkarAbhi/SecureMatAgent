"""
SecureMatAgent FastAPI Application.

Endpoints:
  POST   /api/chat                       — run the agent on a query
  GET    /api/health                     — check Ollama + Qdrant health
  POST   /api/ingest                     — trigger ingestion pipeline
  GET    /api/sessions/{session_id}/history — conversation history
  DELETE /api/sessions/{session_id}      — clear session memory
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (ChatRequest, ChatResponse, DeleteSessionResponse,
                            HealthResponse, HealthStatus, IngestRequest,
                            IngestResponse, SessionHistoryMessage,
                            SessionHistoryResponse, TraceEventModel,
                            TraceLogResponse, TraceSummaryItem)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_trace_summary(session_id: str) -> List[TraceSummaryItem]:
    """Fetch the trace log for session_id and build a condensed summary list."""
    try:
        from src.observability.local_tracer import get_trace_log

        events = get_trace_log(session_id)
        return [
            TraceSummaryItem(
                step=i + 1,
                event_type=ev.event_type,
                name=ev.name,
                latency_ms=ev.latency_ms,
            )
            for i, ev in enumerate(events)
        ]
    except Exception as exc:
        logger.warning("Could not build trace summary for %s: %s", session_id, exc)
        return []


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SecureMatAgent API",
    description=(
        "A RAG-powered agent that answers questions about materials science "
        "and laboratory cybersecurity using a local LLM (Ollama) and vector "
        "search (Qdrant)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health helpers
# ---------------------------------------------------------------------------


def _ollama_url() -> str:
    """Resolve Ollama URL — honours LOCAL_DEV env var."""
    local_dev = os.getenv("LOCAL_DEV", "false").lower() in ("true", "1", "yes")
    if local_dev:
        return "http://localhost:11434"
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _qdrant_url() -> str:
    local_dev = os.getenv("LOCAL_DEV", "false").lower() in ("true", "1", "yes")
    if local_dev:
        return "http://localhost:6333"
    host = os.getenv("QDRANT_HOST", "localhost")
    port = os.getenv("QDRANT_PORT", "6333")
    return f"http://{host}:{port}"


async def _check_ollama() -> HealthStatus:
    url = f"{_ollama_url()}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return HealthStatus(
                    status="ok", message=f"Reachable at {_ollama_url()}"
                )
            return HealthStatus(
                status="degraded",
                message=f"HTTP {resp.status_code} from {url}",
            )
    except Exception as exc:
        return HealthStatus(status="error", message=str(exc))


async def _check_qdrant() -> HealthStatus:
    url = f"{_qdrant_url()}/healthz"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return HealthStatus(
                    status="ok", message=f"Reachable at {_qdrant_url()}"
                )
            return HealthStatus(
                status="degraded",
                message=f"HTTP {resp.status_code} from {url}",
            )
    except Exception as exc:
        return HealthStatus(status="error", message=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Check liveness of Ollama and Qdrant services."""
    ollama_status = await _check_ollama()
    qdrant_status = await _check_qdrant()

    if ollama_status.status == "ok" and qdrant_status.status == "ok":
        overall = "ok"
    elif ollama_status.status == "error" and qdrant_status.status == "error":
        overall = "error"
    else:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        ollama=ollama_status,
        qdrant=qdrant_status,
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["agent"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Submit a question to SecureMatAgent.

    The agent searches the knowledge base, runs tools as needed,
    and returns a structured response with sources and reasoning metadata.
    """
    logger.info(
        "POST /api/chat  session=%s  query=%r", request.session_id, request.query[:80]
    )

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _run_agent() -> Dict[str, Any]:
        from src.agent.run import ask

        return ask(request.query, session_id=request.session_id)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = await loop.run_in_executor(pool, _run_agent)

    if result.get("answer", "").startswith("Agent error:"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["answer"],
        )

    trace_summary = _build_trace_summary(request.session_id)

    return ChatResponse(
        answer=result["answer"],
        tools_used=result["tools_used"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
        session_id=request.session_id,
        trace_summary=trace_summary if trace_summary else None,
    )


@app.post("/api/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Trigger the document ingestion pipeline on a given directory.

    Loads, chunks, embeds, and stores documents into Qdrant.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    logger.info("POST /api/ingest  data_dir=%r", request.data_dir)

    def _run_ingest() -> Dict[str, Any]:
        from src.ingestion.pipeline import run_ingestion

        return run_ingestion(request.data_dir)

    loop = asyncio.get_event_loop()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(pool, _run_ingest)
        return IngestResponse(
            status="ok",
            total_files=result["total_files"],
            total_chunks=result["total_chunks"],
            collection_size=result["collection_size"],
            domain_breakdown=result["domain_breakdown"],
            message=(
                f"Ingested {result['total_files']} file(s) → "
                f"{result['total_chunks']} chunks → "
                f"{result['collection_size']} vectors"
            ),
        )
    except Exception as exc:
        logger.error("Ingestion error: %s", exc, exc_info=True)
        return IngestResponse(status="error", message=str(exc))


@app.get(
    "/api/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    tags=["sessions"],
)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    """Return the conversation history for a session."""
    try:
        from src.agent.memory import get_memory

        checkpointer = get_memory()
        config = {"configurable": {"thread_id": session_id}}

        # get_tuple returns the latest checkpoint or None
        checkpoint_tuple = checkpointer.get_tuple(config)
        messages: List[SessionHistoryMessage] = []

        if checkpoint_tuple is not None:
            checkpoint = checkpoint_tuple.checkpoint
            raw_messages = checkpoint.get("channel_values", {}).get("messages", [])
            for msg in raw_messages:
                cls_name = msg.__class__.__name__
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                if cls_name == "HumanMessage":
                    messages.append(
                        SessionHistoryMessage(role="human", content=content)
                    )
                elif cls_name == "AIMessage":
                    if content.strip():
                        messages.append(
                            SessionHistoryMessage(role="ai", content=content)
                        )
                elif cls_name == "ToolMessage":
                    messages.append(
                        SessionHistoryMessage(
                            role="tool",
                            content=content,
                            name=getattr(msg, "name", None),
                        )
                    )

        return SessionHistoryResponse(
            session_id=session_id,
            messages=messages,
            message_count=len(messages),
        )
    except Exception as exc:
        logger.error("History error for session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {exc}",
        )


@app.get(
    "/api/traces/{session_id}",
    response_model=TraceLogResponse,
    tags=["observability"],
)
async def get_trace_log(session_id: str) -> TraceLogResponse:
    """
    Return the full local trace log for a session.

    Each event records an LLM call, tool call, agent action, or error with
    timestamp, latency, and I/O preview.  Requires local tracing (default).
    """
    try:
        from src.observability.local_tracer import get_trace_log as _get_log

        events = _get_log(session_id)
        return TraceLogResponse(
            session_id=session_id,
            events=[
                TraceEventModel(
                    timestamp=ev.timestamp,
                    event_type=ev.event_type,
                    name=ev.name,
                    input=ev.input,
                    output=ev.output,
                    latency_ms=ev.latency_ms,
                    tokens_used=ev.tokens_used,
                )
                for ev in events
            ],
            event_count=len(events),
        )
    except Exception as exc:
        logger.error(
            "Trace log error for session %s: %s", session_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve trace log: {exc}",
        )


@app.delete(
    "/api/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    tags=["sessions"],
)
async def delete_session(session_id: str) -> DeleteSessionResponse:
    """Clear conversation memory for a session."""
    try:
        from src.agent.memory import clear_session_memory

        cleared = clear_session_memory(session_id)
        return DeleteSessionResponse(
            session_id=session_id,
            cleared=cleared,
            message=(
                f"Session '{session_id}' cleared."
                if cleared
                else f"Session '{session_id}' not found (nothing to clear)."
            ),
        )
    except Exception as exc:
        logger.error("Delete session error %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Entry point (for direct `python -m src.api.main`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
