"""
Document loader for SecureMatAgent ingestion pipeline.

Supports PDF, TXT, and MD files. Adds metadata (source, file_type,
ingestion_timestamp, domain) to each loaded document.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain keyword sets (all lowercase for case-insensitive matching)
# ---------------------------------------------------------------------------
_CYBER_KEYWORDS = {
    "nist",
    "security",
    "cyber",
    "access control",
    "vulnerability",
    "cve",
    "compliance",
    "incident",
}
_MATERIALS_KEYWORDS = {
    "perovskite",
    "xrd",
    "diffraction",
    "lattice",
    "crystal",
    "ceramic",
    "battery",
    "cathode",
    "msds",
    "hazard",
}


def _detect_domain(filename: str, content_preview: str) -> str:
    """Return 'cybersecurity', 'materials_science', or 'general'."""
    haystack = (filename + " " + content_preview).lower()

    cyber_hits = sum(1 for kw in _CYBER_KEYWORDS if kw in haystack)
    mat_hits = sum(1 for kw in _MATERIALS_KEYWORDS if kw in haystack)

    if cyber_hits > mat_hits:
        return "cybersecurity"
    if mat_hits > cyber_hits:
        return "materials_science"
    if cyber_hits == mat_hits and cyber_hits > 0:
        return "cybersecurity"  # tie-break to cybersecurity
    return "general"


def _load_pdf(file_path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as exc:
        logger.warning("PyPDFLoader failed for %s: %s", file_path, exc)
        return []


def _load_text(file_path: str) -> List[Document]:
    try:
        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
        return loader.load()
    except Exception as exc:
        logger.warning("TextLoader failed for %s: %s", file_path, exc)
        return []


def load_documents(data_dir: str) -> List[Document]:
    """
    Load all PDF, TXT, and MD documents from *data_dir*.

    Each returned Document has metadata:
      - source: absolute file path
      - filename: basename
      - file_type: "pdf" | "txt" | "md"
      - ingestion_timestamp: ISO-8601 UTC
      - domain: "cybersecurity" | "materials_science" | "general"
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise ValueError(f"data_dir does not exist: {data_dir}")

    timestamp = datetime.now(tz=timezone.utc).isoformat()
    docs: List[Document] = []
    skipped = 0

    for entry in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, entry)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(entry)[1].lower().lstrip(".")
        if ext not in {"pdf", "txt", "md"}:
            continue

        logger.debug("Loading: %s", entry)

        if ext == "pdf":
            raw_docs = _load_pdf(file_path)
            file_type = "pdf"
        else:
            raw_docs = _load_text(file_path)
            file_type = ext  # "txt" or "md"

        if not raw_docs:
            logger.warning("No content extracted from %s — skipping", entry)
            skipped += 1
            continue

        # Build content preview from first doc's page_content
        preview = raw_docs[0].page_content[:500] if raw_docs else ""
        domain = _detect_domain(entry, preview)

        for doc in raw_docs:
            doc.metadata.update(
                {
                    "source": file_path,
                    "filename": entry,
                    "file_type": file_type,
                    "ingestion_timestamp": timestamp,
                    "domain": domain,
                }
            )
            docs.append(doc)

    logger.info(
        "Loaded %d document pages from %d files (%d skipped) in %s",
        len(docs),
        len(docs) - skipped,  # approximate unique files
        skipped,
        data_dir,
    )
    return docs
