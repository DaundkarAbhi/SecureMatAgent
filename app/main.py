"""
SecureMatAgent FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload --port 8000

In Docker (via entrypoint.sh):
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from config.settings import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== SecureMatAgent API starting ===")
    logger.info(f"Ollama  : {settings.ollama_base_url}  model={settings.ollama_model}")
    logger.info(
        f"Qdrant  : {settings.qdrant_url}  collection={settings.collection_name}"
    )
    logger.info(f"LocalDev: {settings.local_dev}")
    yield
    logger.info("=== SecureMatAgent API shutting down ===")


app = FastAPI(
    title="SecureMatAgent API",
    description="Agentic RAG for cybersecurity-aware scientific research intelligence.",
    version="1.0.0",
    lifespan=lifespan,
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

app.include_router(router, prefix="/api/v1")


# ------------------------------------------------------------------ #
# Root redirect
# ------------------------------------------------------------------ #


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "SecureMatAgent API. See /docs for usage."}
