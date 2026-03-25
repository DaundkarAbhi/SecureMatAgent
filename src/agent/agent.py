"""
SecureMatAgent — LangGraph ReAct agent wiring LLM, tools, retriever, and memory.

LangGraph 1.x / LangChain 1.x edition.

Entry point:
    agent = create_agent()   # compiled LangGraph graph (singleton per process)
    config = get_session_memory(session_id)
    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)

The agent uses Mistral 7B's *tool-calling* (function-calling) interface — far
more reliable on modern Ollama builds than text-based ReAct scratchpads.

System prompt is injected via the ``prompt`` parameter of ``create_react_agent``.
A sliding-window hook keeps only the last K conversation turns in the LLM
context to prevent prompt-bloat.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, SystemMessage

from src.agent.llm import get_llm
from src.agent.memory import get_memory, get_window_size
from src.agent.retriever_tool import document_search
from src.observability.logger import get_logger
from src.observability.tracing import setup_tracing
from src.tools import get_all_tools

# Callbacks are attached per-invocation in src/agent/run.py so that each
# call gets a session-scoped LocalTracer.  setup_tracing() is called here
# at agent creation time to configure the backend (LangSmith or local).
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are SecureMatAgent, an AI research assistant specializing in materials "
    "science with cybersecurity awareness. You help researchers query scientific "
    "literature, calculate material properties, extract structured data, check "
    "data integrity, and stay informed about security best practices for research "
    "labs.\n\n"
    "TOOL USE RULES — follow these strictly:\n"
    "1. For ANY question about materials, crystal structures, XRD, properties, "
    "lab protocols, or scientific data: ALWAYS call document_search first. "
    "Never answer from memory — your training data may be wrong or outdated.\n"
    "2. For ANY numerical calculation (Bragg angles, density, lattice parameters, "
    "unit conversions): ALWAYS call materials_calculator. Never compute inline.\n"
    "3. If document_search returns no relevant results, say so explicitly — "
    "do not hallucinate sources or invent filenames.\n"
    "4. Always cite the source filename from document_search results in your answer."
)

# ---------------------------------------------------------------------------
# Sliding-window prompt hook
# ---------------------------------------------------------------------------


def _make_prompt_hook(system_prompt: str, window_k: int):
    """
    Return a callable that LangGraph calls before every LLM invocation.

    It prepends the system message and trims the message list to the last
    ``window_k`` human/AI turns (2 × K messages) to enforce the memory window.

    Args:
        system_prompt: Plain-text system instructions for the agent.
        window_k:      Number of full turns (human + assistant) to keep.
    """
    sys_msg = SystemMessage(content=system_prompt)

    def _hook(state: Dict[str, Any]) -> List[BaseMessage]:
        messages: List[BaseMessage] = state.get("messages", [])

        # Separate system messages from conversation messages
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        # Keep last window_k turns (each turn = 1 human + 1 AI + N tool messages)
        # Rough approximation: keep the last window_k * 3 messages
        max_msgs = window_k * 3
        trimmed = non_system[-max_msgs:] if len(non_system) > max_msgs else non_system

        return [sys_msg] + trimmed

    return _hook


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_agent():
    """
    Build and return a compiled LangGraph ReAct agent.

    The agent is stateless itself — session memory is injected at invocation
    time via the LangGraph config dict (see ``get_session_memory``).

    Tools bound (5 total):
      1. document_search   — Qdrant knowledge base retriever
      2. materials_calculator — Bragg, density, lattice, unit conversion
      3. web_search        — DuckDuckGo web search
      4. data_anomaly_checker — statistical + range anomaly detection
      5. data_extractor    — structured data extraction from knowledge base

    Returns:
        Compiled LangGraph ``CompiledStateGraph`` ready for ``.invoke()``.
    """
    from langgraph.prebuilt import create_react_agent

    # Configure tracing backend (LangSmith if API key present, else local)
    mode = setup_tracing()
    logger.info("Agent tracing mode: %s", mode)

    llm = get_llm()
    checkpointer = get_memory()
    window_k = get_window_size()

    tools = [document_search] + get_all_tools()

    logger.info(
        "Creating SecureMatAgent | tools=%s | window_k=%d",
        [t.name for t in tools],
        window_k,
    )

    prompt_hook = _make_prompt_hook(_SYSTEM_PROMPT, window_k)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt_hook,
        checkpointer=checkpointer,
        debug=False,
    )

    return agent
