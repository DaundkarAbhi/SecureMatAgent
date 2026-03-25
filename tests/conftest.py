"""
Shared pytest fixtures for SecureMatAgent test suite.

All external services (Qdrant, Ollama/ChatOllama) are mocked so that the
full suite runs without Docker, without network access, and without Ollama.
"""

from __future__ import annotations

import os
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Force test-safe environment variables BEFORE any application module import
# ---------------------------------------------------------------------------
os.environ.setdefault("LOCAL_DEV", "true")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("COLLECTION_NAME", "test_collection")


# ---------------------------------------------------------------------------
# Settings fixture — returns a predictable Settings object
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mock_settings():
    """Return a Settings-like namespace with test-safe values."""
    # Clear the lru_cache so test settings take effect
    from config.settings import Settings, get_settings

    get_settings.cache_clear()

    settings = Settings(
        local_dev=True,
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:7b",
        ollama_temperature=0.1,
        ollama_timeout=30,
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="test_collection",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_device="cpu",
        chunk_size=512,
        chunk_overlap=50,
        top_k=3,
        memory_window=3,
    )
    return settings


# ---------------------------------------------------------------------------
# Mock documents — 5 realistic Document objects
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_documents() -> list[Document]:
    """Return five realistic Document objects covering both domains."""
    return [
        Document(
            page_content=(
                "X-ray diffraction analysis of BaTiO3 perovskite shows a tetragonal "
                "crystal structure at room temperature with lattice parameters a=3.992 Å "
                "and c=4.036 Å. The (110) reflection appears at 2θ=31.5 degrees when "
                "using Cu Kα radiation (λ=1.5406 Å)."
            ),
            metadata={
                "source": "/data/documents/perovskite_XRD.pdf",
                "filename": "perovskite_XRD.pdf",
                "file_type": "pdf",
                "domain": "materials_science",
                "ingestion_timestamp": "2025-01-01T00:00:00+00:00",
            },
        ),
        Document(
            page_content=(
                "NIST SP 800-171 Revision 3 outlines 110 security requirements for "
                "protecting Controlled Unclassified Information (CUI) in nonfederal "
                "systems. Access control requirements include limiting system access to "
                "authorized users and controlling the flow of CUI."
            ),
            metadata={
                "source": "/data/documents/NIST_800_171.pdf",
                "filename": "NIST_800_171.pdf",
                "file_type": "pdf",
                "domain": "cybersecurity",
                "ingestion_timestamp": "2025-01-01T00:00:00+00:00",
            },
        ),
        Document(
            page_content=(
                "Safety Data Sheet for Barium Titanate (BaTiO3). CAS Number: 12047-27-7. "
                "Hazard classification: Eye irritation Category 2. Personal protective "
                "equipment: wear dust mask and safety glasses. Storage: keep in sealed "
                "container away from moisture."
            ),
            metadata={
                "source": "/data/documents/sds_BaTiO3.txt",
                "filename": "sds_BaTiO3.txt",
                "file_type": "txt",
                "domain": "materials_science",
                "ingestion_timestamp": "2025-01-01T00:00:00+00:00",
            },
        ),
        Document(
            page_content=(
                "LiFePO4 cathode material for lithium-ion batteries exhibits an olivine "
                "crystal structure with space group Pnma. Theoretical capacity is "
                "170 mAh/g. The lattice parameters are a=10.33 Å, b=6.01 Å, c=4.69 Å. "
                "XRD patterns confirm phase-pure synthesis after calcination at 700°C."
            ),
            metadata={
                "source": "/data/documents/arxiv_battery_cathode.pdf",
                "filename": "arxiv_battery_cathode.pdf",
                "file_type": "pdf",
                "domain": "materials_science",
                "ingestion_timestamp": "2025-01-01T00:00:00+00:00",
            },
        ),
        Document(
            page_content=(
                "Incident response procedures for research laboratories must include "
                "immediate isolation of affected systems, notification of the security "
                "officer within one hour, and documentation of the incident timeline. "
                "CVE vulnerability scanning should be performed quarterly per NIST "
                "cybersecurity framework guidelines."
            ),
            metadata={
                "source": "/data/documents/lab_security_protocol.md",
                "filename": "lab_security_protocol.md",
                "file_type": "md",
                "domain": "cybersecurity",
                "ingestion_timestamp": "2025-01-01T00:00:00+00:00",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Temporary data directory with sample files
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_data_dir() -> Generator[str, None, None]:
    """Create a temp directory with sample TXT and MD files for loader tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Cybersecurity text file
        nist_file = os.path.join(tmpdir, "NIST_800_171.txt")
        with open(nist_file, "w", encoding="utf-8") as f:
            f.write(
                "NIST SP 800-171 security requirements for protecting CUI.\n"
                "Access control: limit system access to authorized users.\n"
                "Incident response: establish an operational incident-handling capability.\n"
                "Vulnerability scanning: periodically scan for vulnerabilities in systems.\n"
            )

        # Materials science text file
        xrd_file = os.path.join(tmpdir, "perovskite_XRD.txt")
        with open(xrd_file, "w", encoding="utf-8") as f:
            f.write(
                "XRD analysis of perovskite BaTiO3 crystal structure.\n"
                "Diffraction peaks observed at 2theta = 22.1, 31.5, 38.9 degrees.\n"
                "Lattice parameter a = 3.992 Angstroms for cubic phase.\n"
            )

        # Materials markdown file
        lab_proto = os.path.join(tmpdir, "synthesis_protocol.md")
        with open(lab_proto, "w", encoding="utf-8") as f:
            f.write(
                "# Ceramic Synthesis Protocol\n\n"
                "## Materials\n- BaTiO3 powder (ceramic, 99.9% purity)\n"
                "- Ethanol (solvent)\n\n"
                "## Procedure\n1. Weigh 5 g of BaTiO3 powder.\n"
                "2. Calcine at 700°C for 4 hours.\n"
            )

        # Unsupported file type — should be skipped
        unsupported = os.path.join(tmpdir, "spreadsheet.csv")
        with open(unsupported, "w", encoding="utf-8") as f:
            f.write("sample,density\nBaTiO3,6.02\n")

        yield tmpdir


# ---------------------------------------------------------------------------
# In-memory Qdrant client fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def qdrant_client_inmemory():
    """Return a Qdrant QdrantClient backed by in-memory storage."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    return client


# ---------------------------------------------------------------------------
# Session-scoped embedding model (loads once for the entire test session)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def embedding_model():
    """Load HuggingFace all-MiniLM-L6-v2 once for the whole test session."""
    from langchain_huggingface import HuggingFaceEmbeddings

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return model


# ---------------------------------------------------------------------------
# Mock ChatOllama that returns predictable responses
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_llm():
    """
    Return a MagicMock that behaves like a ChatOllama / LLM.

    .invoke() returns a fake AIMessage with predictable content.
    """
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Mock LLM response for testing.")
    llm.bind_tools.return_value = llm  # support tool binding
    return llm


# ---------------------------------------------------------------------------
# Patch get_settings() for the entire test session
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_settings(mock_settings):
    """
    Autouse: patch config.settings.get_settings everywhere so no module
    inadvertently loads production .env values.
    """
    with patch("config.settings.get_settings", return_value=mock_settings):
        yield
