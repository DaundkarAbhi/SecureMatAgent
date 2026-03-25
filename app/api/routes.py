"""
FastAPI route definitions for SecureMatAgent.

Endpoints:
  GET  /health           — liveness probe
  GET  /ready            — readiness probe (checks Qdrant + Ollama)
  POST /chat             — single-turn or multi-turn chat with the agent
  POST /ingest           — trigger document ingestion
  DELETE /memory         — clear conversation memory
"""

from __future__ import annotations

from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from app.agent.rag_agent import SecureMatAgent
from app.ingestion.ingest import ingest
from config.settings import get_settings

router = APIRouter()
settings = get_settings()

# Shared agent instance (lazy-init on first /chat request)
_agent: Optional[SecureMatAgent] = None


def _get_agent() -> SecureMatAgent:
    global _agent
    if _agent is None:
        _agent = SecureMatAgent(settings)
    return _agent


# ------------------------------------------------------------------ #
# Schemas
# ------------------------------------------------------------------ #


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    session_id: Optional[str] = Field(
        default=None, description="Reserved for future multi-session support."
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: Optional[str] = None


class IngestRequest(BaseModel):
    data_dir: Optional[str] = Field(
        default=None, description="Override the default DATA_DIR."
    )


class IngestResponse(BaseModel):
    chunks_upserted: int
    message: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    ollama: str
    qdrant: str


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #


@router.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    return {"status": "ok"}


@router.get("/ready", response_model=ReadyResponse, tags=["Ops"])
async def ready():
    ollama_status = "ok"
    qdrant_status = "ok"

    async with httpx.AsyncClient(timeout=5) as client:
        # Ollama check
        try:
            r = await client.get(settings.ollama_api_tags_url)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(settings.ollama_model in m for m in models):
                ollama_status = (
                    f"warn: model '{settings.ollama_model}' not found in {models}"
                )
        except Exception as exc:
            ollama_status = f"error: {exc}"

        # Qdrant check
        try:
            r = await client.get(f"{settings.qdrant_url}/healthz")
            r.raise_for_status()
        except Exception as exc:
            qdrant_status = f"error: {exc}"

    overall = "ok" if (ollama_status == "ok" and qdrant_status == "ok") else "degraded"
    return {"status": overall, "ollama": ollama_status, "qdrant": qdrant_status}


@router.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(req: ChatRequest):
    try:
        agent = _get_agent()
        answer = agent.chat(req.message)
        return ChatResponse(answer=answer, session_id=req.session_id)
    except Exception as exc:
        logger.error(f"/chat error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )


@router.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def trigger_ingest(req: IngestRequest):
    try:
        count = ingest(data_dir=req.data_dir)
        return IngestResponse(
            chunks_upserted=count,
            message=(
                f"Successfully upserted {count} chunks."
                if count > 0
                else "No documents ingested."
            ),
        )
    except Exception as exc:
        logger.error(f"/ingest error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )


@router.delete("/memory", tags=["Agent"])
async def clear_memory():
    agent = _get_agent()
    agent.reset_memory()
    return {"message": "Conversation memory cleared."}
