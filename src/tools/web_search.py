"""
WebSearchTool — DuckDuckGo search for real-time / current information.

Returns the top-3 results as title + snippet + URL.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Use when you need current information about CVEs, security advisories, \
recent publications, or real-time data not in the document store."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Error: duckduckgo-search package is not installed. Run: pip install duckduckgo-search"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as exc:
        return f"Search failed: {exc}"

    if not results:
        return f"No results found for: {query!r}"

    lines: list[str] = [f"Search results for: {query!r}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        body = r.get("body", r.get("snippet", "No snippet"))
        href = r.get("href", r.get("url", "No URL"))
        lines.append(f"[{i}] {title}\n    {body}\n    URL: {href}")

    return "\n\n".join(lines)
