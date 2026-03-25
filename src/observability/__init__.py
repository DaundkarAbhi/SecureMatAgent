"""
SecureMatAgent observability package.

Provides local callback tracing (default) with optional LangSmith remote tracing.
"""

from src.observability.local_tracer import LocalTracer, get_trace_log
from src.observability.logger import get_logger
from src.observability.tracing import setup_tracing

__all__ = ["setup_tracing", "LocalTracer", "get_trace_log", "get_logger"]
