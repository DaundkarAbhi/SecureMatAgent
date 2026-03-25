"""
Local callback-based tracer for SecureMatAgent.

Captures all LangChain/LangGraph lifecycle events (LLM calls, tool calls,
agent reasoning) and stores them in-memory per session_id.

Optional: set TRACE_TO_FILE=true to also append events to traces/<session_id>.jsonl

Usage:
    tracer = LocalTracer(session_id="abc123")
    # Pass to agent invocation:
    agent.invoke({"messages": [...]}, config={..., "callbacks": [tracer]})

    # Later, retrieve events:
    events = get_trace_log("abc123")
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback as tb_module
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TraceEvent dataclass
# ---------------------------------------------------------------------------


@dataclass
class TraceEvent:
    timestamp: str
    event_type: str  # llm_start | llm_end | tool_start | tool_end |
    # agent_action | agent_finish | chain_error
    name: str
    input: Optional[str] = None
    output: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None


# ---------------------------------------------------------------------------
# Module-level in-memory store  { session_id -> [TraceEvent, ...] }
# ---------------------------------------------------------------------------

_trace_store: Dict[str, List[TraceEvent]] = {}


def get_trace_log(session_id: str) -> List[TraceEvent]:
    """Return all recorded TraceEvents for the given session_id (newest last)."""
    return list(_trace_store.get(session_id, []))


def clear_trace_log(session_id: str) -> None:
    """Remove all trace events for a session."""
    _trace_store.pop(session_id, None)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: Optional[str], limit: int = 500) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    return s[:limit] + "…" if len(s) > limit else s


# ---------------------------------------------------------------------------
# LocalTracer — BaseCallbackHandler subclass
# ---------------------------------------------------------------------------


class LocalTracer(BaseCallbackHandler):
    """
    LangChain callback handler that captures agent trace events per session.

    Thread-safety note: CPython's GIL protects the list.append() calls, so
    concurrent sessions using the shared _trace_store dict are safe under
    CPython with the standard ThreadPoolExecutor used by the API.
    """

    def __init__(self, session_id: str = "default") -> None:
        super().__init__()
        self.session_id = session_id
        self._start_times: Dict[str, float] = {}  # run_id_str -> perf_counter
        self._run_names: Dict[str, str] = {}  # run_id_str -> tool/model name

        self._trace_to_file: bool = os.getenv("TRACE_TO_FILE", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        if self._trace_to_file:
            os.makedirs("traces", exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rid(self, run_id: Optional[UUID]) -> str:
        return str(run_id) if run_id else "unknown"

    def _elapsed(self, run_id_str: str) -> float:
        return round(
            (
                time.perf_counter()
                - self._start_times.pop(run_id_str, time.perf_counter())
            )
            * 1_000,
            1,
        )

    def _store(self, event: TraceEvent) -> None:
        if self.session_id not in _trace_store:
            _trace_store[self.session_id] = []
        _trace_store[self.session_id].append(event)

        logger.debug(
            "TRACE [%s] %-14s | %-20s | latency=%s ms",
            self.session_id,
            event.event_type,
            event.name,
            f"{event.latency_ms:.1f}" if event.latency_ms is not None else "—",
        )

        if self._trace_to_file:
            path = os.path.join("traces", f"{self.session_id}.jsonl")
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(event)) + "\n")
            except OSError as exc:
                logger.warning("Could not write trace file %s: %s", path, exc)

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        self._start_times[rid] = time.perf_counter()

        # Model name: last element of the id list e.g. ["langchain", "chat_models", "ollama", "ChatOllama"]
        model_name = (serialized.get("id") or ["unknown"])[-1]
        self._run_names[rid] = model_name

        prompt_preview = _truncate(prompts[0] if prompts else "")
        self._store(
            TraceEvent(
                timestamp=_now_iso(),
                event_type="llm_start",
                name=model_name,
                input=prompt_preview,
            )
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        latency_ms = self._elapsed(rid)
        model_name = self._run_names.pop(rid, "llm")

        # Token usage (provider-dependent)
        tokens: Optional[int] = None
        if response.llm_output:
            usage = response.llm_output.get("token_usage") or response.llm_output.get(
                "usage", {}
            )
            if isinstance(usage, dict):
                tokens = usage.get("total_tokens") or usage.get("completion_tokens")

        # Text output preview
        text_out: Optional[str] = None
        if response.generations:
            first_gen = response.generations[0]
            if first_gen:
                text_out = _truncate(getattr(first_gen[0], "text", "") or "")

        self._store(
            TraceEvent(
                timestamp=_now_iso(),
                event_type="llm_end",
                name=model_name,
                output=text_out,
                latency_ms=latency_ms,
                tokens_used=tokens,
            )
        )

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        self._start_times[rid] = time.perf_counter()
        tool_name = serialized.get("name", "unknown_tool")
        self._run_names[rid] = tool_name

        self._store(
            TraceEvent(
                timestamp=_now_iso(),
                event_type="tool_start",
                name=tool_name,
                input=_truncate(input_str),
            )
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        latency_ms = self._elapsed(rid)
        tool_name = self._run_names.pop(rid, "tool")

        self._store(
            TraceEvent(
                timestamp=_now_iso(),
                event_type="tool_end",
                name=tool_name,
                output=_truncate(str(output) if output is not None else ""),
                latency_ms=latency_ms,
            )
        )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        latency_ms = self._elapsed(rid)
        tool_name = self._run_names.pop(rid, "tool")

        self._store(
            TraceEvent(
                timestamp=_now_iso(),
                event_type="tool_end",
                name=tool_name,
                output=f"ERROR: {error}",
                latency_ms=latency_ms,
            )
        )
        logger.warning(
            "TRACE tool_error [%s] %s: %s", self.session_id, tool_name, error
        )

    # ------------------------------------------------------------------
    # Agent callbacks
    # ------------------------------------------------------------------

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        thought = _truncate(getattr(action, "log", "") or "")
        tool = getattr(action, "tool", "") or "agent"
        self._store(
            TraceEvent(
                timestamp=_now_iso(),
                event_type="agent_action",
                name=tool,
                input=thought,
            )
        )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        output = getattr(finish, "return_values", {}) or {}
        output_str = _truncate(str(output))
        self._store(
            TraceEvent(
                timestamp=_now_iso(),
                event_type="agent_finish",
                name="agent",
                output=output_str,
            )
        )

    # ------------------------------------------------------------------
    # Error callbacks
    # ------------------------------------------------------------------

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        formatted_tb = _truncate(tb_module.format_exc(), limit=1000)
        self._store(
            TraceEvent(
                timestamp=_now_iso(),
                event_type="chain_error",
                name="chain",
                output=formatted_tb,
            )
        )
        logger.error("TRACE chain_error [%s]: %s", self.session_id, error)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        latency_ms = self._elapsed(rid)
        self._store(
            TraceEvent(
                timestamp=_now_iso(),
                event_type="chain_error",
                name="llm",
                output=f"LLM ERROR: {error}",
                latency_ms=latency_ms,
            )
        )
        logger.error("TRACE llm_error [%s]: %s", self.session_id, error)
