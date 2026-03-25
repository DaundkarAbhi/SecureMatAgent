"""
Document chunker for SecureMatAgent ingestion pipeline.

Uses RecursiveCharacterTextSplitter with settings from config.
All source metadata is preserved; chunk_index is added to each chunk.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings

logger = logging.getLogger(__name__)


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split *docs* into smaller chunks.

    Each chunk's metadata inherits all fields from its parent document
    and gains a ``chunk_index`` field (0-based within that source page/doc).

    Returns the flat list of all chunks across all input documents.
    """
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,  # adds 'start_index' to metadata for traceability
    )

    chunks: List[Document] = []

    for doc in docs:
        try:
            split_docs = splitter.split_documents([doc])
        except Exception as exc:
            logger.warning(
                "Failed to split document %s: %s",
                doc.metadata.get("filename", "<unknown>"),
                exc,
            )
            continue

        for idx, chunk in enumerate(split_docs):
            chunk.metadata["chunk_index"] = idx
            chunks.append(chunk)

    logger.info(
        "Chunked %d source pages → %d chunks (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
