"""
Custom LangChain tools available to the SecureMatAgent.

Tools:
  - VectorStoreRetrieverTool  : semantic search over ingested corpus
  - DuckDuckGoSearchTool      : live web search
  - SympyCalculatorTool       : safe symbolic / numeric math evaluation
  - CVELookupTool             : lightweight CVE query via NVD public API (no key needed)
"""

from __future__ import annotations

import re
from typing import Optional

import httpx
from duckduckgo_search import DDGS
from langchain.tools import BaseTool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sympy import SympifyError, sympify

from config.settings import get_settings

# ------------------------------------------------------------------ #
# Input schemas
# ------------------------------------------------------------------ #


class RetrieverInput(BaseModel):
    query: str = Field(
        description="The natural language query to search the knowledge base."
    )


class SearchInput(BaseModel):
    query: str = Field(description="Web search query string.")
    max_results: int = Field(default=5, description="Max search results to return.")


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Mathematical expression to evaluate (Python/SymPy syntax)."
    )


class CVEInput(BaseModel):
    keyword: str = Field(
        description="CVE ID (e.g. CVE-2024-1234) or keyword to look up in NVD."
    )


# ------------------------------------------------------------------ #
# Tools
# ------------------------------------------------------------------ #


class VectorStoreRetrieverTool(BaseTool):
    """Semantic search over the SecureMatAgent vector store (Qdrant)."""

    name: str = "knowledge_base_search"
    description: str = (
        "Search the ingested scientific and cybersecurity document corpus. "
        "Use this for questions about ingested papers, reports, or internal knowledge. "
        "Input: a natural language query string."
    )
    args_schema: type[BaseModel] = RetrieverInput

    # These are set post-construction via _build_retriever_tool()
    _qdrant_store: Optional[Qdrant] = None
    _top_k: int = 5

    def _run(self, query: str) -> str:
        if self._qdrant_store is None:
            return "Knowledge base not initialised."
        results = self._qdrant_store.similarity_search(query, k=self._top_k)
        if not results:
            return "No relevant documents found in the knowledge base."
        parts = []
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] Source: {src}\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)

    async def _arun(self, query: str) -> str:
        return self._run(query)


class DuckDuckGoSearchTool(BaseTool):
    """Live web search powered by DuckDuckGo (no API key required)."""

    name: str = "web_search"
    description: str = (
        "Search the live web for recent cybersecurity news, CVE details, or scientific findings "
        "not present in the local knowledge base. Input: search query string."
    )
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No web results found."
            lines = []
            for r in results:
                lines.append(
                    f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\n{r.get('body', '')}"
                )
            return "\n\n---\n\n".join(lines)
        except Exception as exc:
            logger.warning(f"DuckDuckGo search failed: {exc}")
            return f"Web search failed: {exc}"

    async def _arun(self, query: str, max_results: int = 5) -> str:
        return self._run(query, max_results)


class SympyCalculatorTool(BaseTool):
    """Safe symbolic / numeric math evaluator using SymPy."""

    name: str = "calculator"
    description: str = (
        "Evaluate mathematical expressions safely. Supports arithmetic, algebra, "
        "trigonometry, logarithms, etc. Input must be a valid Python/SymPy expression."
    )
    args_schema: type[BaseModel] = CalculatorInput

    # Disallow dangerous builtins
    _BLACKLIST = re.compile(r"\b(import|exec|eval|open|os|sys|subprocess)\b")

    def _run(self, expression: str) -> str:
        if self._BLACKLIST.search(expression):
            return "Error: expression contains disallowed keywords."
        try:
            result = sympify(expression, evaluate=True)
            return str(result)
        except SympifyError as exc:
            return f"Math error: {exc}"
        except Exception as exc:
            return f"Evaluation error: {exc}"

    async def _arun(self, expression: str) -> str:
        return self._run(expression)


class CVELookupTool(BaseTool):
    """Query the NVD (National Vulnerability Database) public API — no key needed."""

    name: str = "cve_lookup"
    description: str = (
        "Look up CVE details from the NIST National Vulnerability Database. "
        "Input: a CVE ID like 'CVE-2024-1234' or a keyword/product name."
    )
    args_schema: type[BaseModel] = CVEInput

    _NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    def _run(self, keyword: str) -> str:
        cve_pattern = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
        params: dict = {}
        if cve_pattern.match(keyword.strip()):
            params["cveId"] = keyword.strip().upper()
        else:
            params["keywordSearch"] = keyword.strip()
            params["resultsPerPage"] = 5

        try:
            resp = httpx.get(self._NVD_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            vulns = data.get("vulnerabilities", [])
            if not vulns:
                return "No CVEs found for that query."
            parts = []
            for v in vulns[:5]:
                cve = v.get("cve", {})
                cve_id = cve.get("id", "N/A")
                descs = cve.get("descriptions", [])
                desc_en = next(
                    (d["value"] for d in descs if d.get("lang") == "en"),
                    "No description.",
                )
                severity = (
                    cve.get("metrics", {})
                    .get("cvssMetricV31", [{}])[0]
                    .get("cvssData", {})
                    .get("baseSeverity", "N/A")
                )
                parts.append(f"ID: {cve_id}\nSeverity: {severity}\n{desc_en}")
            return "\n\n---\n\n".join(parts)
        except httpx.HTTPStatusError as exc:
            return f"NVD API error {exc.response.status_code}: {exc}"
        except Exception as exc:
            logger.warning(f"CVE lookup failed: {exc}")
            return f"CVE lookup failed: {exc}"

    async def _arun(self, keyword: str) -> str:
        return self._run(keyword)


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #


def build_tools(settings=None) -> list[BaseTool]:
    """Construct and return all agent tools, wiring up shared resources."""
    if settings is None:
        settings = get_settings()

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    qdrant_store = Qdrant(
        client=client,
        collection_name=settings.collection_name,
        embeddings=embeddings,
    )

    retriever_tool = VectorStoreRetrieverTool()
    retriever_tool._qdrant_store = qdrant_store
    retriever_tool._top_k = settings.top_k

    return [
        retriever_tool,
        DuckDuckGoSearchTool(),
        SympyCalculatorTool(),
        CVELookupTool(),
    ]
