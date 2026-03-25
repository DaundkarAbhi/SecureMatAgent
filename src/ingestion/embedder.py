"""
Embedding model provider for SecureMatAgent ingestion pipeline.

Uses HuggingFaceEmbeddings with sentence-transformers/all-MiniLM-L6-v2
(384-dimensional vectors). The model is loaded once and reused via a
module-level singleton.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton — None until first call to get_embedding_model()
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return the shared HuggingFaceEmbeddings instance.

    The model is loaded on first call and cached for all subsequent calls.
    Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
    """
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        logger.info(
            "Loading embedding model '%s' on device '%s' ...",
            settings.embedding_model,
            settings.embedding_device,
        )
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded.")
    return _embedding_model
