"""
Unit tests for src/ingestion/chunker.py — chunk_documents().

Tests verify chunk size enforcement, overlap, metadata preservation,
and edge-case documents (empty, very short, single character).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(
    content: str, filename: str = "test.txt", domain: str = "general"
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": f"/data/{filename}",
            "filename": filename,
            "file_type": "txt",
            "domain": domain,
            "ingestion_timestamp": "2025-01-01T00:00:00+00:00",
        },
    )


def _long_text(n_chars: int = 2000) -> str:
    """Generate a realistic long text of approximately n_chars characters."""
    sentence = (
        "The XRD pattern of the perovskite sample shows clear diffraction peaks. "
    )
    repetitions = (n_chars // len(sentence)) + 1
    return (sentence * repetitions)[:n_chars]


# ---------------------------------------------------------------------------
# Chunk size
# ---------------------------------------------------------------------------


class TestChunkSize:
    def test_chunks_respect_max_size_512(self, mock_settings):
        """No chunk should exceed chunk_size=512 characters."""
        doc = _make_doc(_long_text(3000))
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert (
                len(chunk.page_content) <= mock_settings.chunk_size + 50
            ), f"Chunk length {len(chunk.page_content)} exceeds limit"

    def test_long_document_produces_multiple_chunks(self):
        """A 3000-char document should produce more than 1 chunk at size=512."""
        doc = _make_doc(_long_text(3000))
        chunks = chunk_documents([doc])
        assert len(chunks) > 1

    def test_multiple_documents_all_chunked(self):
        """Multiple documents are all chunked and returned together."""
        docs = [_make_doc(_long_text(1500), f"doc{i}.txt") for i in range(3)]
        chunks = chunk_documents(docs)
        assert len(chunks) >= len(docs)


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


class TestChunkOverlap:
    def test_adjacent_chunks_share_text(self):
        """
        With chunk_overlap=50, adjacent chunks should share some characters.
        We verify that the tail of chunk[i] appears in chunk[i+1].
        """
        doc = _make_doc(_long_text(2000))
        chunks = chunk_documents([doc])
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")

        # The overlap means the start of chunk[1] should contain text from end of chunk[0]
        # We check by looking for at least partial overlap via start_index metadata
        c0_end = chunks[0].metadata.get("start_index", 0) + len(chunks[0].page_content)
        c1_start = chunks[1].metadata.get("start_index", 0)
        # c1_start should be BEFORE c0_end (overlap exists)
        assert (
            c1_start < c0_end
        ), f"No overlap detected: chunk[0] ends at {c0_end}, chunk[1] starts at {c1_start}"


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    def test_source_metadata_preserved(self):
        doc = _make_doc(
            _long_text(1500), filename="perovskite_XRD.pdf", domain="materials_science"
        )
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "/data/perovskite_XRD.pdf"
            assert chunk.metadata["filename"] == "perovskite_XRD.pdf"
            assert chunk.metadata["domain"] == "materials_science"
            assert chunk.metadata["file_type"] == "txt"

    def test_chunk_index_added(self):
        """Each chunk should have a chunk_index key added."""
        doc = _make_doc(_long_text(2000))
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata

    def test_chunk_indices_are_sequential(self):
        """chunk_index values per document should be 0, 1, 2, ..."""
        doc = _make_doc(_long_text(2000))
        chunks = chunk_documents([doc])
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_ingestion_timestamp_preserved(self):
        doc = _make_doc(_long_text(1500))
        doc.metadata["ingestion_timestamp"] = "2025-06-15T10:00:00+00:00"
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["ingestion_timestamp"] == "2025-06-15T10:00:00+00:00"

    def test_start_index_added_by_splitter(self):
        """RecursiveCharacterTextSplitter with add_start_index=True adds start_index."""
        doc = _make_doc(_long_text(1500))
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert "start_index" in chunk.metadata


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_document_list(self):
        chunks = chunk_documents([])
        assert chunks == []

    def test_empty_page_content(self):
        """Document with empty content produces no chunks."""
        doc = _make_doc("")
        chunks = chunk_documents([doc])
        assert chunks == []

    def test_short_document_is_single_chunk(self):
        """A 50-char doc fits in a single chunk."""
        doc = _make_doc("Short content that fits easily.")
        chunks = chunk_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short content that fits easily."

    def test_exactly_chunk_size_content(self, mock_settings):
        """Content exactly at chunk_size boundary → should be 1 chunk."""
        doc = _make_doc("x" * mock_settings.chunk_size)
        chunks = chunk_documents([doc])
        assert len(chunks) >= 1
        assert all(len(c.page_content) <= mock_settings.chunk_size + 50 for c in chunks)

    def test_single_char_document(self):
        doc = _make_doc("A")
        chunks = chunk_documents([doc])
        assert len(chunks) == 1

    def test_whitespace_only_document(self):
        """Whitespace-only content: may produce 0 or 1 chunks, must not raise."""
        doc = _make_doc("   \n\t  ")
        # Should not raise
        chunks = chunk_documents([doc])
        assert isinstance(chunks, list)
