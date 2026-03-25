"""
SecureMatAgent — Agentic RAG core.

Wraps a LangChain ReAct agent (Mistral 7B via Ollama) with:
  - Conversational memory (windowed)
  - Tool use: knowledge base search, web search, calculator, CVE lookup
  - Streaming support via callbacks
"""

from __future__ import annotations

from typing import Iterator

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from loguru import logger

from app.agent.tools import build_tools
from config.settings import get_settings

# ------------------------------------------------------------------ #
# System / ReAct prompt
# ------------------------------------------------------------------ #

REACT_SYSTEM_PROMPT = """You are SecureMatAgent, an expert AI research assistant specializing in
cybersecurity-aware scientific research intelligence. You help researchers discover, analyze, and
synthesize information from academic papers, security advisories, CVE databases, and the open web.

You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Guidelines:
- Always search the knowledge base first before going to the web.
- When discussing vulnerabilities, cite CVE IDs when available.
- Be concise, accurate, and cite sources in your Final Answer.
- If the knowledge base has no relevant information, say so and use web search.
- For mathematical claims, use the calculator to verify.

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


# ------------------------------------------------------------------ #
# Agent factory
# ------------------------------------------------------------------ #


def build_agent(settings=None) -> AgentExecutor:
    """Build and return a fully initialised AgentExecutor."""
    if settings is None:
        settings = get_settings()

    logger.info(
        f"Initialising LLM: {settings.ollama_model} @ {settings.ollama_base_url}"
    )
    llm = OllamaLLM(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
        timeout=settings.ollama_timeout,
    )

    tools = build_tools(settings)
    logger.info(f"Tools loaded: {[t.name for t in tools]}")

    prompt = PromptTemplate(
        input_variables=[
            "tools",
            "tool_names",
            "chat_history",
            "input",
            "agent_scratchpad",
        ],
        template=REACT_SYSTEM_PROMPT,
    )

    memory = ConversationBufferWindowMemory(
        k=settings.memory_window,
        memory_key="chat_history",
        return_messages=False,
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        early_stopping_method="generate",
    )

    logger.success("SecureMatAgent ready.")
    return executor


# ------------------------------------------------------------------ #
# Convenience wrapper
# ------------------------------------------------------------------ #


class SecureMatAgent:
    """Thin stateful wrapper around AgentExecutor for use by the API and UI."""

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._executor: AgentExecutor | None = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = build_agent(self._settings)
        return self._executor

    def chat(self, user_message: str) -> str:
        """Synchronous chat — returns the Final Answer string."""
        try:
            result = self.executor.invoke({"input": user_message})
            return result.get("output", "No response generated.")
        except Exception as exc:
            logger.error(f"Agent error: {exc}")
            return f"Agent error: {exc}"

    def reset_memory(self) -> None:
        """Clear conversation history."""
        if self._executor:
            self._executor.memory.clear()  # type: ignore[union-attr]
            logger.info("Conversation memory cleared.")
