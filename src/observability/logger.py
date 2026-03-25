"""
Structured JSON logging for SecureMatAgent.

Usage:
    from src.observability.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Request received", extra={"session_id": "abc"})

Log levels:
  DEBUG   — detailed trace events (tool inputs/outputs)
  INFO    — request lifecycle (query received, agent complete)
  WARNING — retries, degraded services, slow responses
  ERROR   — agent failures, exceptions, service outages
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Merge any extra fields passed via logger.info(..., extra={...})
        _skip = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",
        }
        for key, val in record.__dict__.items():
            if key not in _skip and not key.startswith("_"):
                log_obj[key] = val

        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


# Track which loggers we've already configured to avoid duplicate handlers
_configured: set[str] = set()


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a logger with JSON output configured.

    Loggers are only configured once — subsequent calls with the same name
    return the cached instance without adding duplicate handlers.

    Args:
        name:  Logger name, typically ``__name__``.
        level: Minimum log level (default: DEBUG so all levels pass through;
               the root logger / environment can raise the floor).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    log = logging.getLogger(name)

    if name in _configured:
        return log

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    handler.setLevel(level)

    log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False  # don't double-log via root logger

    _configured.add(name)
    return log
