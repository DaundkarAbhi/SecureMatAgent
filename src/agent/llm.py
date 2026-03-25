"""
LLM factory for SecureMatAgent.

Returns a ChatOllama instance configured from settings.
URL resolves automatically: localhost (LOCAL_DEV=true) or host.docker.internal.
"""

from __future__ import annotations

import logging

from langchain_ollama import ChatOllama

from config.settings import get_settings

logger = logging.getLogger(__name__)


def get_llm() -> ChatOllama:
    """
    Build and return a ChatOllama instance.

    URL is resolved from settings:
      - LOCAL_DEV=true  → http://localhost:11434
      - default         → http://host.docker.internal:11434

    Returns:
        ChatOllama configured with model, temperature, and timeout from settings.
    """
    settings = get_settings()

    logger.info(
        "Initialising ChatOllama | model=%s | url=%s | temperature=%.2f | timeout=%ds",
        settings.ollama_model,
        settings.ollama_base_url,
        settings.ollama_temperature,
        settings.ollama_timeout,
    )

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        timeout=settings.ollama_timeout,
        keep_alive="5m",
    )

    return llm
