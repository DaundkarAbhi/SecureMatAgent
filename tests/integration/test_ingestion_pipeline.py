"""
Integration tests for the full ingestion pipeline:
    load_documents → chunk_documents → (embed) → ingest_documents

Uses Qdrant in-memory mode — no Docker, no network required.
Embedding model loads from HuggingFace cache (session-scoped fixture).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

pytestmark = pytest.mark.integration

VECTOR_DIM = 384
TEST_COLLECTION = "test_ingestion"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inmemory_store(collection_name: str, embedding_model):
    """
    Build a QdrantVectorStore backed by an in-memory client.
    Returns (client, store).
    """
    from langchain_qdrant import QdrantVectorStore

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )
    return client, store


# ---------------------------------------------------------------------------
# Stage-by-stage tests
# ---------------------------------------------------------------------------


class TestLoadStage:
    def test_load_produces_documents(self, temp_data_dir):
        from src.ingestion.loader import load_documents

        docs = load_documents(temp_data_dir)
        assert len(docs) >= 3  # NIST txt + XRD txt + synthesis md

    def test_load_has_domain_tags(self, temp_data_dir):
        from src.ingestion.loader import load_documents

        docs = load_documents(temp_data_dir)
        domains = {d.metadata["domain"] for d in docs}
        assert "cybersecurity" in domains
        assert "materials_science" in domains


class TestChunkStage:
    def test_chunk_count_exceeds_doc_count(self, temp_data_dir):
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_documents

        docs = load_documents(temp_data_dir)
        chunks = chunk_documents(docs)
        # At least as many chunks as docs (long docs split into multiple)
        assert len(chunks) >= len(docs)

    def test_chunk_metadata_has_chunk_index(self, temp_data_dir):
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_documents

        docs = load_documents(temp_data_dir)
        chunks = chunk_documents(docs)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_chunk_metadata_preserves_domain(self, temp_data_dir):
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_documents

        docs = load_documents(temp_data_dir)
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "domain" in chunk.metadata
            assert chunk.metadata["domain"] in {
                "cybersecurity",
                "materials_science",
                "general",
            }


class TestEmbedAndStoreStage:
    def test_vectors_are_384_dimensional(self, embedding_model):
        """Verify the embedding model produces 384-dim vectors."""
        vecs = embedding_model.embed_documents(["test sentence"])
        assert len(vecs) == 1
        assert len(vecs[0]) == VECTOR_DIM

    def test_ingest_into_inmemory_qdrant(self, embedding_model, mock_documents):
        """
        Ingest mock_documents into an in-memory Qdrant collection and
        verify the count matches.
        """
        client, store = _make_inmemory_store(TEST_COLLECTION, embedding_model)
        import uuid

        ids = [str(uuid.uuid4()) for _ in mock_documents]
        store.add_documents(documents=mock_documents, ids=ids)

        info = client.get_collection(TEST_COLLECTION)
        assert info.points_count == len(mock_documents)

    def test_retrieved_docs_have_metadata(self, embedding_model, mock_documents):
        """Similarity search returns documents with metadata intact."""
        client, store = _make_inmemory_store(TEST_COLLECTION + "_meta", embedding_model)
        import uuid

        ids = [str(uuid.uuid4()) for _ in mock_documents]
        store.add_documents(documents=mock_documents, ids=ids)

        retriever = store.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke("XRD diffraction perovskite")

        assert len(results) > 0
        for doc in results:
            assert "domain" in doc.metadata
            assert "filename" in doc.metadata

    def test_similarity_search_returns_relevant_docs(
        self, embedding_model, mock_documents
    ):
        """XRD query should return materials_science docs, not cybersecurity."""
        client, store = _make_inmemory_store(TEST_COLLECTION + "_rel", embedding_model)
        import uuid

        ids = [str(uuid.uuid4()) for _ in mock_documents]
        store.add_documents(documents=mock_documents, ids=ids)

        results = store.similarity_search("XRD perovskite crystal structure", k=2)
        domains = [r.metadata.get("domain") for r in results]
        assert "materials_science" in domains


# ---------------------------------------------------------------------------
# Full pipeline (mocked vectorstore connection, real chunking + embedding)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_pipeline_load_chunk_embed_store(
        self, temp_data_dir, embedding_model, mock_settings
    ):
        """
        Full pipeline: load → chunk → embed → store into in-memory Qdrant.
        Mocks the QdrantClient constructor to return an in-memory client.
        """
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_documents

        # Step 1: Load
        docs = load_documents(temp_data_dir)
        assert len(docs) >= 1

        # Step 2: Chunk
        chunks = chunk_documents(docs)
        assert len(chunks) >= len(docs)

        # Step 3+4: Embed + Store into in-memory Qdrant
        client, store = _make_inmemory_store("full_pipeline_test", embedding_model)
        import uuid

        ids = [str(uuid.uuid4()) for _ in chunks]
        store.add_documents(documents=chunks, ids=ids)

        info = client.get_collection("full_pipeline_test")
        assert info.points_count == len(chunks)

    def test_pipeline_domain_breakdown(self, temp_data_dir, embedding_model):
        """Verify both domains appear in the ingested chunks."""
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_documents

        docs = load_documents(temp_data_dir)
        chunks = chunk_documents(docs)

        domain_counts: dict[str, int] = {}
        for c in chunks:
            d = c.metadata.get("domain", "general")
            domain_counts[d] = domain_counts.get(d, 0) + 1

        assert "cybersecurity" in domain_counts
        assert "materials_science" in domain_counts

    def test_pipeline_with_run_ingestion_mocked(self, temp_data_dir, embedding_model):
        """
        Test run_ingestion() with QdrantClient patched to use in-memory mode.

        pipeline.py does `from qdrant_client import QdrantClient` locally, so we
        patch `qdrant_client.QdrantClient` at the package level to intercept it.
        """
        in_memory_client = QdrantClient(":memory:")

        def mock_qdrant_constructor(*args, **kwargs):
            return in_memory_client

        with (
            patch(
                "src.ingestion.vectorstore._get_client", return_value=in_memory_client
            ),
            patch(
                "src.ingestion.vectorstore.get_embedding_model",
                return_value=embedding_model,
            ),
            patch("qdrant_client.QdrantClient", side_effect=mock_qdrant_constructor),
        ):
            from src.ingestion.pipeline import run_ingestion

            result = run_ingestion(temp_data_dir)

        assert result["total_files"] >= 1
        assert result["total_chunks"] >= result["total_files"]
        assert isinstance(result["domain_breakdown"], dict)
        assert len(result["domain_breakdown"]) >= 1
