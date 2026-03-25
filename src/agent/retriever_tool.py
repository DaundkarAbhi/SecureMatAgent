"""
LangChain @tool that wraps the Qdrant retriever.

The tool is named "document_search" and searches the materials science /
cybersecurity knowledge base, returning formatted passages with source
citations.

Optional domain filtering: if the query ends with a filter tag such as
  [domain:cybersecurity]   or   [domain:materials_science]
the tag is stripped from the query and used as a metadata filter.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.tools import tool

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Matches an optional trailing tag like [domain:cybersecurity]
_DOMAIN_TAG_RE = re.compile(r"\[domain:([^\]]+)\]\s*$", re.IGNORECASE)


def _get_retriever(filter_domain: Optional[str] = None):
    """Lazy import to avoid circular deps and heavy init at module load time."""
    from src.ingestion.vectorstore import get_retriever

    settings = get_settings()
    return get_retriever(top_k=settings.top_k, filter_domain=filter_domain)


def _format_results(docs: list) -> str:
    """Format retrieved documents into a readable string with citations."""
    if not docs:
        return "No relevant documents found in the knowledge base."

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", meta.get("filename", "unknown source"))
        # Trim to just the filename for brevity
        if "/" in source or "\\" in source:
            source = re.split(r"[/\\]", source)[-1]
        domain = meta.get("domain", "")
        domain_tag = f" [{domain}]" if domain else ""

        parts.append(
            f"[{i}] Source: {source}{domain_tag}\n" f"{doc.page_content.strip()}"
        )

    return "\n\n".join(parts)


@tool
def document_search(query: str) -> str:
    """Search the materials science and cybersecurity knowledge base. \
Use for questions about material properties, lab protocols, NIST guidelines, \
security frameworks, XRD analysis, crystal structures, or any domain-specific \
information. Returns relevant passages with source citations. \
Optionally append [domain:cybersecurity] or [domain:materials_science] to \
restrict results to a specific domain."""
    try:
        # Parse optional domain filter from query
        domain: Optional[str] = None
        clean_query = query.strip()
        m = _DOMAIN_TAG_RE.search(clean_query)
        if m:
            domain = m.group(1).strip()
            clean_query = _DOMAIN_TAG_RE.sub("", clean_query).strip()
            logger.debug("document_search: domain filter=%s", domain)

        if not clean_query:
            return "Please provide a non-empty search query."

        logger.info("document_search query='%s' domain=%s", clean_query, domain)
        retriever = _get_retriever(filter_domain=domain)
        docs = retriever.invoke(clean_query)

        result = _format_results(docs)
        logger.debug("document_search returned %d chunk(s).", len(docs))
        return result

    except Exception as exc:
        logger.error("document_search error: %s", exc, exc_info=True)
        return f"Knowledge base search failed: {exc}"
