"""
src/tools — LangChain agent tools for SecureMatAgent.

Usage:
    from src.tools import get_all_tools
    tools = get_all_tools()
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .anomaly_checker import data_anomaly_checker
from .calculator import materials_calculator
from .data_extractor import data_extractor
from .web_search import web_search


def get_all_tools() -> list[BaseTool]:
    """Return all registered agent tools."""
    return [
        materials_calculator,
        web_search,
        data_anomaly_checker,
        data_extractor,
    ]


__all__ = [
    "get_all_tools",
    "materials_calculator",
    "web_search",
    "data_anomaly_checker",
    "data_extractor",
]
