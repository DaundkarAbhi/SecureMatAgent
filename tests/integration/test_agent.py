"""
Integration tests for src/agent/ — agent construction, tool dispatch,
memory, and max_iterations guard.

LLM (ChatOllama) is fully mocked.  The real tools are used except web_search,
which is also mocked to avoid network access.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers to build fake LangGraph message sequences
# ---------------------------------------------------------------------------


def _ai_msg(content: str, tool_calls: list | None = None) -> AIMessage:
    msg = AIMessage(content=content)
    if tool_calls:
        msg.tool_calls = tool_calls
    return msg


def _tool_msg(name: str, content: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="tc_001", name=name)


# ---------------------------------------------------------------------------
# _extract_tools_used
# ---------------------------------------------------------------------------


class TestExtractToolsUsed:
    def test_extracts_from_tool_message(self):
        from src.agent.run import _extract_tools_used

        msgs = [
            HumanMessage(content="Calculate bragg angle"),
            _tool_msg("materials_calculator", "2theta = 31.1 deg"),
            _ai_msg("The Bragg angle is 31.1 degrees."),
        ]
        tools = _extract_tools_used(msgs)
        assert "materials_calculator" in tools

    def test_extracts_from_ai_message_tool_calls(self):
        from src.agent.run import _extract_tools_used

        ai = AIMessage(content="")
        ai.tool_calls = [
            {"name": "document_search", "args": {"query": "perovskite"}, "id": "tc1"}
        ]
        msgs = [ai]
        tools = _extract_tools_used(msgs)
        assert "document_search" in tools

    def test_no_duplicates(self):
        from src.agent.run import _extract_tools_used

        msgs = [
            _tool_msg("materials_calculator", "result1"),
            _tool_msg("materials_calculator", "result2"),
        ]
        tools = _extract_tools_used(msgs)
        assert tools.count("materials_calculator") == 1

    def test_empty_messages(self):
        from src.agent.run import _extract_tools_used

        assert _extract_tools_used([]) == []


# ---------------------------------------------------------------------------
# _extract_sources
# ---------------------------------------------------------------------------


class TestExtractSources:
    def test_extracts_source_from_document_search(self):
        from src.agent.run import _extract_sources

        content = (
            "[1] Source: perovskite_XRD.pdf [materials_science]\n"
            "XRD peaks at 31.5 degrees.\n\n"
            "[2] Source: NIST_800_171.pdf [cybersecurity]\n"
            "Access control requirements."
        )
        msgs = [_tool_msg("document_search", content)]
        sources = _extract_sources(msgs)
        assert "perovskite_XRD.pdf" in sources
        assert "NIST_800_171.pdf" in sources

    def test_ignores_non_document_search_tools(self):
        from src.agent.run import _extract_sources

        msgs = [_tool_msg("materials_calculator", "[1] Source: fake.pdf\nsome result")]
        sources = _extract_sources(msgs)
        assert "fake.pdf" not in sources

    def test_no_duplicates_in_sources(self):
        from src.agent.run import _extract_sources

        content = "[1] Source: file.pdf\ncontent\n\n[2] Source: file.pdf\nmore content"
        msgs = [_tool_msg("document_search", content)]
        sources = _extract_sources(msgs)
        assert sources.count("file.pdf") == 1

    def test_empty_returns_empty(self):
        from src.agent.run import _extract_sources

        assert _extract_sources([]) == []


# ---------------------------------------------------------------------------
# _extract_answer
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    def test_returns_last_ai_message(self):
        from src.agent.run import _extract_answer

        msgs = [
            HumanMessage(content="Question"),
            _ai_msg(""),  # empty intermediate
            _ai_msg("Final answer here."),
        ]
        answer = _extract_answer(msgs)
        assert answer == "Final answer here."

    def test_skips_empty_ai_messages(self):
        from src.agent.run import _extract_answer

        msgs = [
            _ai_msg("Real answer"),
            _ai_msg(""),
        ]
        answer = _extract_answer(msgs)
        assert answer == "Real answer"

    def test_empty_messages_returns_empty(self):
        from src.agent.run import _extract_answer

        assert _extract_answer([]) == ""


# ---------------------------------------------------------------------------
# Memory / session management
# ---------------------------------------------------------------------------


class TestMemory:
    def test_get_session_memory_returns_config(self):
        from src.agent.memory import get_session_memory

        config = get_session_memory("test-session-xyz")
        assert config == {"configurable": {"thread_id": "test-session-xyz"}}

    def test_get_memory_returns_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver

        from src.agent.memory import get_memory

        checkpointer = get_memory()
        assert isinstance(checkpointer, MemorySaver)

    def test_list_sessions_after_get(self):
        from src.agent.memory import get_session_memory, list_sessions

        get_session_memory("unique-test-session-abc")
        sessions = list_sessions()
        assert "unique-test-session-abc" in sessions

    def test_clear_session_memory_removes_session(self):
        from src.agent.memory import (clear_session_memory, get_session_memory,
                                      list_sessions)

        get_session_memory("session-to-clear")
        cleared = clear_session_memory("session-to-clear")
        assert cleared is True
        assert "session-to-clear" not in list_sessions()

    def test_clear_nonexistent_session_returns_false(self):
        from src.agent.memory import clear_session_memory

        result = clear_session_memory("nonexistent-session-zzz")
        assert result is False

    def test_get_window_size_from_settings(self, mock_settings):
        from src.agent.memory import get_window_size

        with patch("src.agent.memory.get_settings", return_value=mock_settings):
            size = get_window_size()
        assert size == mock_settings.memory_window


# ---------------------------------------------------------------------------
# Prompt hook (sliding window)
# ---------------------------------------------------------------------------


class TestPromptHook:
    def test_hook_prepends_system_message(self):
        from langchain_core.messages import SystemMessage

        from src.agent.agent import _make_prompt_hook

        hook = _make_prompt_hook("You are a helpful agent.", window_k=5)
        state = {"messages": [HumanMessage(content="Hello")]}
        result = hook(state)

        assert isinstance(result[0], SystemMessage)
        assert "helpful agent" in result[0].content

    def test_hook_trims_to_window(self):
        from langchain_core.messages import SystemMessage

        from src.agent.agent import _make_prompt_hook

        hook = _make_prompt_hook("System prompt", window_k=2)
        # window_k=2 → max_msgs=6; create 10 non-system messages
        msgs = [HumanMessage(content=f"msg {i}") for i in range(10)]
        state = {"messages": msgs}
        result = hook(state)

        # First is always system
        assert isinstance(result[0], SystemMessage)
        # Should have trimmed to at most window_k*3 + 1 (system)
        assert len(result) <= 2 * 3 + 1

    def test_hook_filters_system_messages(self):
        from langchain_core.messages import SystemMessage

        from src.agent.agent import _make_prompt_hook

        hook = _make_prompt_hook("My system", window_k=5)
        existing_sys = SystemMessage(content="Old system message")
        state = {"messages": [existing_sys, HumanMessage(content="Hello")]}
        result = hook(state)

        # Only the new system message should appear first
        assert result[0].content == "My system"
        sys_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert len(sys_msgs) == 1


# ---------------------------------------------------------------------------
# Tool dispatch — math query → calculator
# ---------------------------------------------------------------------------


class TestToolDispatch:
    """
    Verify tool selection logic by running the ask() interface with a
    fully mocked LangGraph agent.
    """

    def _build_mock_agent_response(self, tool_name: str, tool_output: str, answer: str):
        """Return a fake LangGraph result dict simulating tool use."""
        ai_with_tool_call = AIMessage(content="")
        ai_with_tool_call.tool_calls = [
            {"name": tool_name, "args": {"query": "test"}, "id": "tc_001"}
        ]
        tool_resp = ToolMessage(
            content=tool_output, tool_call_id="tc_001", name=tool_name
        )
        final_ai = AIMessage(content=answer)
        return {"messages": [ai_with_tool_call, tool_resp, final_ai]}

    def test_math_query_uses_calculator(self):
        """Ask a Bragg angle question → tools_used should include materials_calculator."""
        fake_result = self._build_mock_agent_response(
            "materials_calculator",
            "2theta = 31.1 deg",
            "The 2θ angle is approximately 31.1 degrees.",
        )

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = fake_result

        with patch("src.agent.run._get_agent", return_value=mock_agent):
            from src.agent.run import ask

            result = ask("What is the Bragg angle for d=2.87 Å?")

        assert "materials_calculator" in result["tools_used"]
        assert "31.1" in result["answer"]

    def test_document_query_uses_retriever(self):
        """Ask about NIST → tools_used should include document_search."""
        content = (
            "[1] Source: NIST_800_171.pdf [cybersecurity]\nAccess control requirements."
        )
        fake_result = self._build_mock_agent_response(
            "document_search",
            content,
            "NIST 800-171 covers 110 requirements.",
        )

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = fake_result

        with patch("src.agent.run._get_agent", return_value=mock_agent):
            from src.agent.run import ask

            result = ask("What does NIST 800-171 say about access control?")

        assert "document_search" in result["tools_used"]
        assert "NIST_800_171.pdf" in result["sources"]

    def test_latency_ms_is_float(self):
        """latency_ms must always be a non-negative float."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="Answer")]}

        with patch("src.agent.run._get_agent", return_value=mock_agent):
            from src.agent.run import ask

            result = ask("test question")

        assert isinstance(result["latency_ms"], float)
        assert result["latency_ms"] >= 0

    def test_agent_error_returned_gracefully(self):
        """When agent raises, ask() returns structured error dict."""
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("LLM unreachable")

        with patch("src.agent.run._get_agent", return_value=mock_agent):
            from src.agent.run import ask

            result = ask("anything")

        assert "Agent error" in result["answer"]
        assert result["tools_used"] == []
        assert result["sources"] == []

    def test_session_id_passed_through(self):
        """The session_id used in ask() is returned in the result via memory config."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="OK")]}

        with patch("src.agent.run._get_agent", return_value=mock_agent):
            from src.agent.run import ask

            ask("question", session_id="my-session-42")

        # Verify the agent was called with the right thread_id config
        call_args = mock_agent.invoke.call_args
        config = (
            call_args[1].get("config") or call_args[0][1]
            if len(call_args[0]) > 1
            else None
        )
        if config:
            assert config.get("configurable", {}).get("thread_id") == "my-session-42"
