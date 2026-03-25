"""
DataExtractorTool — LangChain agent tool for structured data extraction.

Two-phase approach:
  Phase 1: Retrieve relevant document chunks from Qdrant.
  Phase 2: Send chunks + extraction prompt to Ollama/Mistral and parse the
           JSON response into a typed Pydantic model.

Extraction types (auto-detected from query keywords):
  • material_properties  — crystal system, lattice params, density, band gap …
  • safety               — GHS, hazard statements, first aid, PPE …
  • compliance           — control IDs, titles, descriptions …
  • table                — tabular data with headers and rows
  • general              — key-value findings (fallback)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from config.settings import get_settings
from src.tools.extraction_prompts import (compliance_prompt,
                                          general_extraction_prompt,
                                          material_properties_prompt,
                                          retry_suffix, safety_data_prompt,
                                          table_extraction_prompt)
from src.tools.extraction_schemas import (ComplianceResult,
                                          GeneralKeyValueResult,
                                          MaterialPropertiesResult,
                                          SafetyDataResult, TableResult)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword sets for extraction-type detection
# ---------------------------------------------------------------------------

_MATERIAL_KEYWORDS = frozenset(
    {
        "lattice",
        "crystal",
        "space group",
        "unit cell",
        "density",
        "band gap",
        "melting point",
        "thermal",
        "hardness",
        "modulus",
        "composition",
        "stoichiometry",
        "phase",
        "grain",
        "xrd",
        "diffraction",
        "material properties",
        "properties of",
        "characteriz",
    }
)

_SAFETY_KEYWORDS = frozenset(
    {
        "safety",
        "hazard",
        "ghs",
        "sds",
        "msds",
        "toxic",
        "flammable",
        "ppe",
        "first aid",
        "exposure",
        "carcinogen",
        "corrosive",
        "h-code",
        "p-code",
        "epa",
        "osha",
        "reach",
        "disposal",
        "storage",
        "dangerous",
        "risk phrase",
    }
)

_COMPLIANCE_KEYWORDS = frozenset(
    {
        "compliance",
        "control",
        "nist",
        "iso 27001",
        "soc 2",
        "pci",
        "hipaa",
        "gdpr",
        "requirement",
        "regulation",
        "framework",
        "audit",
        "policy",
        "access control",
        "incident response",
        "risk management",
        "standard",
        "800-53",
        "cmmc",
        "fedramp",
    }
)

_TABLE_KEYWORDS = frozenset(
    {
        "table",
        "tabulate",
        "list all",
        "enumerate",
        "spreadsheet",
        "columns",
        "rows",
        "matrix",
        "comparison",
        "vs",
        "versus",
        "side by side",
        "data table",
    }
)


def _detect_extraction_type(query: str) -> str:
    """Return one of: material_properties | safety | compliance | table | general."""
    q = query.lower()
    scores = {
        "material_properties": sum(1 for kw in _MATERIAL_KEYWORDS if kw in q),
        "safety": sum(1 for kw in _SAFETY_KEYWORDS if kw in q),
        "compliance": sum(1 for kw in _COMPLIANCE_KEYWORDS if kw in q),
        "table": sum(1 for kw in _TABLE_KEYWORDS if kw in q),
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "general"
    return best


def _format_chunks(docs) -> str:
    """Convert a list of LangChain Documents into a context string."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = (
            doc.metadata.get("source") or doc.metadata.get("filename") or f"chunk_{i}"
        )
        parts.append(f"[Source {i}: {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def _collect_sources(docs) -> list:
    seen: list = []
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("filename") or "unknown"
        if src not in seen:
            seen.append(src)
    return seen


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


@tool
def data_extractor(query: str) -> str:
    """Extract structured data from documents: material properties, tables, \
chemical compositions, lattice parameters, safety information, compliance \
requirements. Use when the user asks to extract, list, tabulate, pull out, \
or needs structured data rather than narrative answers."""
    settings = get_settings()

    # ------------------------------------------------------------------
    # Phase 1 — Retrieve relevant chunks from Qdrant
    # ------------------------------------------------------------------
    try:
        from src.ingestion.vectorstore import \
            get_retriever  # local import avoids circular deps

        retriever = get_retriever(top_k=6)
        docs = retriever.invoke(query)
    except Exception as exc:
        logger.error("DataExtractorTool: retrieval failed — %s", exc)
        return f"Extraction failed during retrieval: {exc}"

    if not docs:
        return "No relevant document chunks found in the knowledge base for this query."

    context = _format_chunks(docs)
    sources = _collect_sources(docs)

    # ------------------------------------------------------------------
    # Phase 2 — Build prompt and call Ollama/Mistral
    # ------------------------------------------------------------------
    extraction_type = _detect_extraction_type(query)
    logger.debug("DataExtractorTool: detected extraction type '%s'", extraction_type)

    prompt_builders = {
        "material_properties": material_properties_prompt,
        "safety": safety_data_prompt,
        "compliance": compliance_prompt,
        "table": table_extraction_prompt,
        "general": general_extraction_prompt,
    }
    prompt_text = prompt_builders[extraction_type](query, context)

    try:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
        )
    except Exception as exc:
        logger.error("DataExtractorTool: LLM init failed — %s", exc)
        return f"Extraction failed: could not initialise LLM. {exc}"

    def call_llm(prompt: str) -> str:
        try:
            response = llm.invoke(prompt)
            # ChatOllama returns an AIMessage
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            return content
        except Exception as exc:
            raise RuntimeError(f"LLM call failed: {exc}") from exc

    def make_retry_fn(base_prompt: str):
        """Return a callable that appends the error and retries once."""

        def retry(error_msg: str) -> str:
            logger.warning(
                "DataExtractorTool: retrying after parse error: %s", error_msg[:120]
            )
            retried_prompt = base_prompt + retry_suffix(error_msg)
            return call_llm(retried_prompt)

        return retry

    try:
        raw_output = call_llm(prompt_text)
    except RuntimeError as exc:
        return f"Extraction failed during LLM call: {exc}"

    retry_fn = make_retry_fn(prompt_text)

    # ------------------------------------------------------------------
    # Parse response into typed schema
    # ------------------------------------------------------------------
    parsers = {
        "material_properties": MaterialPropertiesResult,
        "safety": SafetyDataResult,
        "compliance": ComplianceResult,
        "table": TableResult,
        "general": GeneralKeyValueResult,
    }

    schema_cls = parsers[extraction_type]
    result = schema_cls.parse_llm_output(raw_output, retry_fn=retry_fn)

    # Merge retriever-sourced filenames if the LLM left sources empty
    if not result.sources:
        result.sources = sources

    return result.to_markdown()
