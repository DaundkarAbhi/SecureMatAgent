"""
Conversation memory for SecureMatAgent (LangGraph 1.x / LangChain 1.x).

LangGraph manages conversation history via a *checkpointer* that is keyed by
a ``thread_id`` supplied in the config dict at invocation time.  Each unique
``thread_id`` maps to an independent conversation session.

Public API (mirrors the original ConversationBufferWindowMemory design where
possible, adapted to the LangGraph checkpointer model):

    checkpointer = get_memory()               # shared MemorySaver instance
    config       = get_session_memory(sid)    # {"configurable": {"thread_id": sid}}
    ok           = clear_session_memory(sid)  # wipe a session's history
    ids          = list_sessions()            # active session IDs
"""

from __future__ import annotations

import logging
from typing import Dict, List

from langgraph.checkpoint.memory import MemorySaver

from config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared in-process checkpointer (one per application lifetime)
# ---------------------------------------------------------------------------
_checkpointer: MemorySaver = MemorySaver()

# Track which session IDs have been used (MemorySaver has no listing API)
_active_sessions: Dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_memory() -> MemorySaver:
    """
    Return the shared LangGraph MemorySaver checkpointer.

    Pass this to ``create_react_agent(..., checkpointer=get_memory())``.
    The window size (K) is enforced by the agent's ``prompt`` hook — see
    ``src.agent.agent.create_agent``.
    """
    return _checkpointer


def get_session_memory(session_id: str = "default") -> Dict[str, dict]:
    """
    Return the LangGraph config dict for *session_id*.

    Usage::

        config = get_session_memory("user-42")
        result = agent.invoke({"messages": [...]}, config=config)

    Args:
        session_id: Unique identifier for the conversation session.

    Returns:
        ``{"configurable": {"thread_id": session_id}}``
    """
    _active_sessions[session_id] = True
    return {"configurable": {"thread_id": session_id}}


def clear_session_memory(session_id: str) -> bool:
    """
    Remove stored checkpoint data for *session_id*.

    Returns True if the session existed and was removed, False otherwise.
    """
    existed = _active_sessions.pop(session_id, None) is not None

    # MemorySaver stores data in its internal `.storage` dict keyed by
    # (thread_id, ...).  Remove all entries belonging to this thread.
    storage = getattr(_checkpointer, "storage", {})
    keys_to_delete = [
        k for k in storage if isinstance(k, tuple) and k and k[0] == session_id
    ]
    for k in keys_to_delete:
        del storage[k]

    if existed or keys_to_delete:
        logger.info("Cleared memory for session '%s'.", session_id)
        return True
    return False


def list_sessions() -> List[str]:
    """Return IDs of all sessions that have been started."""
    return list(_active_sessions.keys())


def get_window_size() -> int:
    """Return the configured conversation window size (K turns)."""
    return get_settings().memory_window
