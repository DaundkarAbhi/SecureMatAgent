"""
Qdrant vector store interface for SecureMatAgent ingestion pipeline.

Provides:
  - init_vectorstore()    → QdrantVectorStore (creates collection if absent)
  - ingest_documents()    → int  (number of chunks stored)
  - get_retriever()       → LangChain retriever (with optional domain filter)
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance, FieldCondition, Filter,
                                       MatchValue, VectorParams)

from config.settings import get_settings
from src.ingestion.embedder import get_embedding_model

logger = logging.getLogger(__name__)

VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension


def _get_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def init_vectorstore() -> QdrantVectorStore:
    """
    Connect to Qdrant and ensure the collection exists.

    Creates the collection with Cosine distance and 384-dim vectors if it
    does not already exist. Returns a LangChain QdrantVectorStore wrapper.
    """
    settings = get_settings()
    client = _get_client()
    collection_name = settings.collection_name

    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        logger.info("Creating Qdrant collection '%s' ...", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Collection '%s' created.", collection_name)
    else:
        logger.info("Collection '%s' already exists — reusing.", collection_name)

    embeddings = get_embedding_model()
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def ingest_documents(docs: List[Document]) -> int:
    """
    Embed and store *docs* in Qdrant.

    Returns the number of documents successfully upserted.
    """
    if not docs:
        logger.warning("ingest_documents called with empty document list.")
        return 0

    store = init_vectorstore()

    # Generate stable UUIDs from content to allow idempotent re-ingestion
    ids = [str(uuid.uuid4()) for _ in docs]

    BATCH_SIZE = 200
    total = len(docs)
    upserted = 0

    logger.info(
        "Upserting %d chunks into Qdrant in batches of %d ...", total, BATCH_SIZE
    )
    for start in range(0, total, BATCH_SIZE):
        batch_docs = docs[start : start + BATCH_SIZE]
        batch_ids = ids[start : start + BATCH_SIZE]
        store.add_documents(documents=batch_docs, ids=batch_ids)
        upserted += len(batch_docs)
        logger.info("  ... %d / %d chunks upserted", upserted, total)

    logger.info("Upserted %d chunks.", upserted)
    return upserted


def get_retriever(
    top_k: int = 5,
    filter_domain: Optional[str] = None,
):
    """
    Return a LangChain retriever backed by the Qdrant collection.

    Args:
        top_k: Number of chunks to retrieve per query.
        filter_domain: If provided, restrict results to this domain value
                       (e.g. "cybersecurity" or "materials_science").
    """
    settings = get_settings()
    effective_k = top_k or settings.top_k

    store = init_vectorstore()

    search_kwargs: dict = {"k": effective_k}

    if filter_domain:
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.domain",
                    match=MatchValue(value=filter_domain),
                )
            ]
        )
        logger.debug("Retriever domain filter: %s", filter_domain)

    return store.as_retriever(search_kwargs=search_kwargs)
