"""
Unit tests for src/ingestion/loader.py — load_documents() and domain detection.

PDF loading is mocked (no real PDFs required).  TXT / MD files use real temp
files created by the temp_data_dir fixture.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.ingestion.loader import _detect_domain, load_documents

pytestmark = pytest.mark.unit


# ===========================================================================
# _detect_domain — keyword-based domain detection
# ===========================================================================


class TestDetectDomain:
    def test_nist_filename_is_cybersecurity(self):
        domain = _detect_domain("NIST_800_171.pdf", "")
        assert domain == "cybersecurity"

    def test_xrd_filename_is_materials_science(self):
        domain = _detect_domain("perovskite_XRD.pdf", "")
        assert domain == "materials_science"

    def test_security_in_content_is_cybersecurity(self):
        domain = _detect_domain(
            "report.txt", "This document covers security compliance requirements"
        )
        assert domain == "cybersecurity"

    def test_crystal_in_content_is_materials_science(self):
        domain = _detect_domain(
            "data.txt", "Crystal structure analysis of perovskite BaTiO3"
        )
        assert domain == "materials_science"

    def test_both_keywords_cyber_wins_tiebreak(self):
        """Equal hits default to 'cybersecurity' as tie-break."""
        domain = _detect_domain("nist_perovskite.txt", "")
        # 'nist' -> cyber=1, 'perovskite' -> mat=1  → tie → cybersecurity
        assert domain == "cybersecurity"

    def test_no_keywords_is_general(self):
        domain = _detect_domain(
            "readme.txt", "This is a general document with no keywords"
        )
        assert domain == "general"

    def test_battery_is_materials_science(self):
        domain = _detect_domain("lithium_battery.pdf", "cathode material for battery")
        assert domain == "materials_science"

    def test_cve_is_cybersecurity(self):
        domain = _detect_domain(
            "advisory.txt", "CVE-2024-1234 vulnerability in OpenSSL"
        )
        assert domain == "cybersecurity"

    def test_msds_is_materials_science(self):
        domain = _detect_domain("msds_batio3.txt", "hazard safety data sheet")
        assert domain == "materials_science"

    def test_case_insensitive(self):
        """Domain detection must be case-insensitive."""
        domain = _detect_domain("PEROVSKITE_STUDY.PDF", "CRYSTAL LATTICE DIFFRACTION")
        assert domain == "materials_science"

    @pytest.mark.parametrize(
        "filename,content,expected",
        [
            ("NIST_800_171.pdf", "", "cybersecurity"),
            ("perovskite_XRD.pdf", "", "materials_science"),
            ("sds_chemicals.txt", "msds hazard material", "materials_science"),
            ("lab_protocol.md", "ceramic synthesis crystal", "materials_science"),
            ("incident_report.txt", "security vulnerability cve", "cybersecurity"),
            ("misc_notes.txt", "random generic content here", "general"),
        ],
    )
    def test_parametrize_domain(self, filename, content, expected):
        domain = _detect_domain(filename, content)
        assert (
            domain == expected
        ), f"{filename!r} + content → got {domain!r}, expected {expected!r}"


# ===========================================================================
# load_documents — file loading
# ===========================================================================


class TestLoadDocuments:
    def test_invalid_dir_raises_value_error(self):
        with pytest.raises(ValueError, match="does not exist"):
            load_documents("/nonexistent/path/xyz")

    def test_loads_txt_files(self, temp_data_dir):
        docs = load_documents(temp_data_dir)
        assert len(docs) > 0

    def test_loads_md_files(self, temp_data_dir):
        docs = load_documents(temp_data_dir)
        md_docs = [d for d in docs if d.metadata.get("file_type") == "md"]
        assert len(md_docs) > 0

    def test_skips_unsupported_csv(self, temp_data_dir):
        """CSV files should be silently skipped (not in {pdf, txt, md})."""
        docs = load_documents(temp_data_dir)
        csv_docs = [d for d in docs if d.metadata.get("filename", "").endswith(".csv")]
        assert len(csv_docs) == 0

    def test_metadata_keys_present(self, temp_data_dir):
        """Every loaded doc must have all required metadata keys."""
        docs = load_documents(temp_data_dir)
        required_keys = {
            "source",
            "filename",
            "file_type",
            "ingestion_timestamp",
            "domain",
        }
        for doc in docs:
            missing = required_keys - set(doc.metadata.keys())
            assert (
                not missing
            ), f"Missing keys {missing} in {doc.metadata.get('filename')}"

    def test_nist_txt_domain_cybersecurity(self, temp_data_dir):
        """NIST_800_171.txt should be tagged as cybersecurity."""
        docs = load_documents(temp_data_dir)
        nist_docs = [
            d for d in docs if "NIST_800_171" in d.metadata.get("filename", "")
        ]
        assert len(nist_docs) > 0
        assert nist_docs[0].metadata["domain"] == "cybersecurity"

    def test_perovskite_xrd_domain_materials(self, temp_data_dir):
        """perovskite_XRD.txt should be tagged as materials_science."""
        docs = load_documents(temp_data_dir)
        xrd_docs = [
            d for d in docs if "perovskite_XRD" in d.metadata.get("filename", "")
        ]
        assert len(xrd_docs) > 0
        assert xrd_docs[0].metadata["domain"] == "materials_science"

    def test_source_is_absolute_path(self, temp_data_dir):
        """source metadata must be an absolute path."""
        docs = load_documents(temp_data_dir)
        for doc in docs:
            assert os.path.isabs(
                doc.metadata["source"]
            ), f"source is not absolute: {doc.metadata['source']}"

    def test_pdf_loading_mocked(self, temp_data_dir):
        """
        Mock PyPDFLoader to simulate a PDF being present and loaded.
        Verifies the pipeline handles PDF metadata correctly.
        """
        import shutil

        fake_pdf = os.path.join(temp_data_dir, "NIST_security_framework.pdf")
        # Copy an existing txt as fake pdf (loader is mocked)
        shutil.copy(
            os.path.join(temp_data_dir, "NIST_800_171.txt"),
            fake_pdf,
        )

        fake_doc = Document(
            page_content="NIST cybersecurity framework content from PDF",
            metadata={"source": fake_pdf},
        )

        with patch("src.ingestion.loader.PyPDFLoader") as MockLoader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [fake_doc]
            MockLoader.return_value = mock_loader_instance

            docs = load_documents(temp_data_dir)

        pdf_docs = [d for d in docs if d.metadata.get("file_type") == "pdf"]
        assert len(pdf_docs) >= 1
        assert pdf_docs[0].metadata["domain"] == "cybersecurity"

    def test_unsupported_file_type_logged_as_skip(self, temp_data_dir, caplog):
        """Loading a dir with a CSV should not log an error, just skip silently."""
        with caplog.at_level(logging.DEBUG, logger="src.ingestion.loader"):
            docs = load_documents(temp_data_dir)
        # No ERROR level entries for the csv skip
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        csv_errors = [r for r in errors if "csv" in r.message.lower()]
        assert len(csv_errors) == 0

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        docs = load_documents(str(tmp_path))
        assert docs == []
