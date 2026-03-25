"""
Ingestion pipeline orchestrator for SecureMatAgent.

Runs the full load → chunk → embed → store sequence and returns a
summary dict with counts at each stage.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.ingestion.vectorstore import ingest_documents, init_vectorstore

logger = logging.getLogger(__name__)


def run_ingestion(data_dir: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline: load → chunk → embed → store.

    Args:
        data_dir: Path to the directory containing source documents.

    Returns:
        {
            "total_files":      int,   # unique source files found
            "total_chunks":     int,   # chunks produced by splitter
            "collection_size":  int,   # total vectors now in Qdrant
            "domain_breakdown": dict,  # {domain: chunk_count}
        }
    """
    # ------------------------------------------------------------------
    # Stage 1: Load
    # ------------------------------------------------------------------
    logger.info("=== Stage 1/3: Loading documents from '%s' ===", data_dir)
    docs = load_documents(data_dir)
    unique_files = len({d.metadata.get("filename") for d in docs})
    logger.info("  Loaded %d pages from %d unique files.", len(docs), unique_files)

    if not docs:
        logger.warning("No documents loaded — aborting pipeline.")
        return {
            "total_files": 0,
            "total_chunks": 0,
            "collection_size": 0,
            "domain_breakdown": {},
        }

    # ------------------------------------------------------------------
    # Stage 2: Chunk
    # ------------------------------------------------------------------
    logger.info("=== Stage 2/3: Chunking documents ===")
    chunks = chunk_documents(docs)
    logger.info("  Produced %d chunks.", len(chunks))

    domain_breakdown: Dict[str, int] = dict(
        Counter(c.metadata.get("domain", "general") for c in chunks)
    )

    # ------------------------------------------------------------------
    # Stage 3: Ingest into Qdrant
    # ------------------------------------------------------------------
    logger.info("=== Stage 3/3: Ingesting chunks into Qdrant ===")
    ingest_documents(chunks)

    # Query collection size after ingestion
    from qdrant_client import QdrantClient

    from config.settings import get_settings

    settings = get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collection_info = client.get_collection(settings.collection_name)
    collection_size = collection_info.points_count or 0

    logger.info(
        "=== Pipeline complete: %d files | %d chunks | %d vectors in collection ===",
        unique_files,
        len(chunks),
        collection_size,
    )

    return {
        "total_files": unique_files,
        "total_chunks": len(chunks),
        "collection_size": collection_size,
        "domain_breakdown": domain_breakdown,
    }
