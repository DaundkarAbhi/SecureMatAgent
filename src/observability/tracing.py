"""
Tracing setup for SecureMatAgent.

Behaviour:
- LANGCHAIN_API_KEY set → LangSmith remote tracing (sets LANGCHAIN_TRACING_V2=true)
- LANGCHAIN_API_KEY not set → local callback tracing only (default, no external calls)

Call setup_tracing() once at application startup.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_tracing_mode: str = "local"  # module-level cache so we only log once


def setup_tracing() -> str:
    """
    Configure the active tracing backend.

    Reads environment variables and configures accordingly.

    Returns:
        "langsmith" if LangSmith is enabled, "local" otherwise.
    """
    global _tracing_mode

    api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()

    if api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        project = os.getenv("LANGCHAIN_PROJECT", "securematagent")
        os.environ["LANGCHAIN_PROJECT"] = project
        _tracing_mode = "langsmith"
        logger.info(
            "Tracing backend: LangSmith (project=%s, endpoint=%s)",
            project,
            os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        )
    else:
        # Explicitly disable LangSmith so no accidental remote calls are made
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        _tracing_mode = "local"
        logger.info(
            "Tracing backend: local callback tracer "
            "(set LANGCHAIN_API_KEY to enable LangSmith)"
        )

    return _tracing_mode


def get_tracing_mode() -> str:
    """Return the currently active tracing mode: 'langsmith' | 'local'."""
    return _tracing_mode
