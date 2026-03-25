"""
Document ingestion pipeline.

Loads PDFs (and plain text) from DATA_DIR, splits them into chunks,
embeds with sentence-transformers, and upserts into Qdrant.

Usage (local dev):
    python -m app.ingestion.ingest
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (DirectoryLoader, PyPDFLoader,
                                                  TextLoader)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config.settings import get_settings


def _build_embeddings(settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _ensure_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        logger.info(
            f"Creating Qdrant collection '{collection_name}' (dim={vector_size})"
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists — upserting.")


def load_documents(data_dir: str) -> List[Document]:
    path = Path(data_dir)
    if not path.exists():
        logger.warning(f"DATA_DIR '{data_dir}' does not exist. No documents loaded.")
        return []

    docs: List[Document] = []

    # PDFs
    pdf_loader = DirectoryLoader(
        str(path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    try:
        docs.extend(pdf_loader.load())
    except Exception as exc:
        logger.warning(f"PDF loading error: {exc}")

    # Plain text / markdown
    for ext in ("*.txt", "*.md"):
        txt_loader = DirectoryLoader(
            str(path),
            glob=f"**/{ext}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        try:
            docs.extend(txt_loader.load())
        except Exception as exc:
            logger.warning(f"Text loading error for {ext}: {exc}")

    logger.info(f"Loaded {len(docs)} raw document pages/sections from '{data_dir}'")
    return docs


def ingest(data_dir: str | None = None) -> int:
    """Run the full ingest pipeline. Returns number of chunks upserted."""
    settings = get_settings()
    data_dir = data_dir or settings.data_dir

    logger.info("=== SecureMatAgent Ingestion Pipeline ===")
    logger.info(f"Qdrant : {settings.qdrant_url}")
    logger.info(f"Collection: {settings.collection_name}")
    logger.info(f"Embed model: {settings.embedding_model}")
    logger.info(f"Data dir: {data_dir}")

    # 1. Load
    raw_docs = load_documents(data_dir)
    if not raw_docs:
        logger.error("No documents found. Aborting ingest.")
        return 0

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info(
        f"Split into {len(chunks)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )

    # 3. Embed
    embeddings = _build_embeddings(settings)
    # Probe embedding dim
    sample_vec = embeddings.embed_query("probe")
    vector_size = len(sample_vec)
    logger.info(f"Embedding dimension: {vector_size}")

    # 4. Qdrant collection setup
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _ensure_collection(client, settings.collection_name, vector_size)

    # 5. Upsert via LangChain wrapper
    qdrant_store = Qdrant(
        client=client,
        collection_name=settings.collection_name,
        embeddings=embeddings,
    )
    qdrant_store.add_documents(chunks)
    logger.success(f"Upserted {len(chunks)} chunks into '{settings.collection_name}'.")
    return len(chunks)


if __name__ == "__main__":
    count = ingest()
    sys.exit(0 if count > 0 else 1)
