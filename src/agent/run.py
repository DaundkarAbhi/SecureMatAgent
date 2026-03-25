"""
High-level ask() interface for SecureMatAgent.

Usage:
    from src.agent.run import ask

    result = ask("What XRD peaks are reported for anatase TiO2?")
    print(result["answer"])
    print(result["sources"])
    print(f"Latency: {result['latency_ms']:.1f} ms")
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from src.observability.local_tracer import LocalTracer
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Module-level agent singleton (created lazily on first call)
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from src.agent.agent import create_agent

        _agent = create_agent()
    return _agent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_tools_used(messages: list) -> List[str]:
    """
    Collect the names of tools invoked during a LangGraph agent run.

    Tool calls are recorded in AIMessage.tool_calls and ToolMessage.name.
    """
    tools: List[str] = []
    for msg in messages:
        # AIMessage with tool_calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                name = (
                    tc.get("name")
                    if isinstance(tc, dict)
                    else getattr(tc, "name", None)
                )
                if name and name not in tools:
                    tools.append(name)
        # ToolMessage (the observation)
        if hasattr(msg, "name") and msg.__class__.__name__ == "ToolMessage":
            name = msg.name
            if name and name not in tools:
                tools.append(name)
    return tools


def _extract_sources(messages: list) -> List[str]:
    """
    Scan ToolMessage observations from document_search and extract filenames.

    Lines formatted as '[N] Source: filename.pdf' are parsed out.
    """
    sources: List[str] = []
    for msg in messages:
        if msg.__class__.__name__ != "ToolMessage":
            continue
        tool_name = getattr(msg, "name", "")
        if tool_name != "document_search":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("[") and "Source:" in line:
                try:
                    source_part = line.split("Source:", 1)[1].strip()
                    # Strip domain tag like [cybersecurity]
                    if "[" in source_part:
                        source_part = source_part[: source_part.index("[")].strip()
                    if source_part and source_part not in sources:
                        sources.append(source_part)
                except IndexError:
                    pass
    return sources


def _extract_answer(messages: list) -> str:
    """Return the content of the last AIMessage (the final answer)."""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage":
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            # Some models return content as a list of dicts
            if isinstance(content, list):
                text_parts = [
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                ]
                joined = " ".join(p for p in text_parts if p).strip()
                if joined:
                    return joined
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ask(
    query: str,
    session_id: str = "default",
) -> Dict[str, Any]:
    """
    Submit a question to SecureMatAgent and return a structured response.

    Args:
        query:      The user's question or instruction.
        session_id: Conversation session identifier.  Calls with the same
                    session_id share memory (last K=5 turns by default).

    Returns:
        dict with keys:
          answer      (str)   — the agent's final answer
          tools_used  (list)  — names of tools invoked during reasoning
          sources     (list)  — source filenames cited from the knowledge base
          latency_ms  (float) — wall-clock time for the full agent run
    """
    from langchain_core.messages import HumanMessage

    from src.agent.memory import get_session_memory

    logger.info("ask() | session=%s | query=%r", session_id, query[:120])
    t0 = time.perf_counter()

    try:
        agent = _get_agent()
        config = get_session_memory(session_id)

        # Attach a session-scoped LocalTracer so all events are captured
        tracer = LocalTracer(session_id=session_id)
        config = {**config, "callbacks": [tracer]}

        raw: Dict[str, Any] = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )

        latency_ms = (time.perf_counter() - t0) * 1_000
        messages = raw.get("messages", [])

        answer = _extract_answer(messages)
        tools_used = _extract_tools_used(messages)
        sources = _extract_sources(messages)

        logger.info(
            "ask() complete | session=%s | latency=%.1f ms | tools=%s | sources=%s",
            session_id,
            latency_ms,
            tools_used,
            sources,
        )

        return {
            "answer": answer,
            "tools_used": tools_used,
            "sources": sources,
            "latency_ms": round(latency_ms, 1),
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1_000
        logger.error(
            "ask() error | session=%s | %.1f ms | %s",
            session_id,
            latency_ms,
            exc,
            exc_info=True,
        )
        return {
            "answer": f"Agent error: {exc}",
            "tools_used": [],
            "sources": [],
            "latency_ms": round(latency_ms, 1),
        }
