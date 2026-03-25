"""
NIST collector — downloads public-domain NIST publications and extracts
relevant chapter ranges to keep file sizes manageable.

All NIST Special Publications are public domain (17 U.S.C. § 105).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import (download_pdf, extract_pdf_pages, get_file_metadata,
                    validate_pdf)

logger = logging.getLogger(__name__)

# ── Publication definitions ────────────────────────────────────────────────
# (output_filename, url, start_page, end_page, description)
#
# Page ranges are approximate — we fall back gracefully if the document is
# shorter than expected.  All page numbers are 1-based.
#
NIST_PUBLICATIONS = [
    {
        "filename": "nist_sp800_171r3_requirements.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-171r3.pdf",
        "start_page": 1,
        "end_page": 60,  # Chapters 1-3 + Appendix A (requirements)
        "domain": "cybersecurity",
        "doc_type": "nist_standard",
        "description": (
            "NIST SP 800-171 Rev 3 — Protecting Controlled Unclassified Information "
            "in Nonfederal Systems. Chapters 1-3 and Appendix A (requirements)."
        ),
    },
    {
        "filename": "nist_sp800_53r5_access_control.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
        "start_page": 1,
        "end_page": 80,  # Intro + AC (Access Control) and SC (System & Comm Protection) families
        "domain": "cybersecurity",
        "doc_type": "nist_standard",
        "description": (
            "NIST SP 800-53 Rev 5 — Security and Privacy Controls. "
            "Introductory chapters plus the Access Control (AC) and "
            "System and Communications Protection (SC) control families."
        ),
    },
    {
        "filename": "nist_csf_v2_core_framework.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
        "start_page": 1,
        "end_page": 32,  # Core framework sections (skip implementation examples)
        "domain": "cybersecurity",
        "doc_type": "nist_framework",
        "description": (
            "NIST Cybersecurity Framework v2.0 — Core framework sections covering "
            "Identify, Protect, Detect, Respond, and Recover functions."
        ),
    },
    {
        "filename": "nist_sp800_88r1_media_sanitization.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-88r1.pdf",
        "start_page": 1,
        "end_page": 40,  # Chapters 1-4 (guidelines)
        "domain": "cybersecurity",
        "doc_type": "nist_standard",
        "description": (
            "NIST SP 800-88 Rev 1 — Guidelines for Media Sanitization. "
            "Chapters 1-4 covering sanitization categories, decision process, "
            "and technology-specific guidance."
        ),
    },
]


def collect(output_dir: Path, verbose: bool = False) -> list[dict[str, Any]]:
    """
    Download each NIST publication, extract relevant pages, validate,
    and return manifest entries.  Idempotent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_nist"
    tmp_dir.mkdir(exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []

    for pub in NIST_PUBLICATIONS:
        final_path = output_dir / pub["filename"]

        # ── Already done? ──────────────────────────────────────────────────
        if final_path.exists() and validate_pdf(final_path):
            meta = get_file_metadata(final_path)
            if meta["pages"] > 0:
                logger.info(
                    "  [skip] %s already exists (%d pages)",
                    pub["filename"],
                    meta["pages"],
                )
                manifest_entries.append(_build_entry(pub, final_path))
                if verbose:
                    print(f"    [skip] {pub['filename']}  ({meta['size_kb']:.0f} KB)")
                continue

        # ── Download full PDF to tmp ────────────────────────────────────────
        tmp_path = tmp_dir / ("_raw_" + pub["filename"])
        logger.info("Downloading %s", pub["url"])

        ok = download_pdf(pub["url"], tmp_path, timeout=120)
        if not ok or not tmp_path.exists():
            logger.warning("  Failed to download %s — skipping", pub["filename"])
            continue

        # ── Extract relevant pages ─────────────────────────────────────────
        logger.info(
            "  Extracting pages %d–%d from %s",
            pub["start_page"],
            pub["end_page"],
            tmp_path.name,
        )
        ok = extract_pdf_pages(tmp_path, final_path, pub["start_page"], pub["end_page"])

        # Clean up raw file regardless of extraction result
        if tmp_path.exists():
            tmp_path.unlink()

        if not ok or not validate_pdf(final_path):
            # Extraction failed — use the full download as fallback
            logger.warning(
                "  Page extraction failed; falling back to full download for %s",
                pub["filename"],
            )
            ok = download_pdf(pub["url"], final_path, timeout=120)
            if not ok:
                logger.error("  Cannot obtain %s — skipping", pub["filename"])
                continue

        meta = get_file_metadata(final_path)
        logger.info(
            "  → %s  (%d pages, %.0f KB)",
            pub["filename"],
            meta["pages"],
            meta["size_kb"],
        )
        if verbose:
            print(
                f"    ✓ {pub['filename']}  ({meta['size_kb']:.0f} KB, {meta['pages']} pages)"
            )

        manifest_entries.append(_build_entry(pub, final_path))

    # Cleanup tmp dir
    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # not empty — leave it

    logger.info(
        "NIST collector finished: %d documents collected", len(manifest_entries)
    )
    return manifest_entries


def _build_entry(pub: dict, path: Path) -> dict[str, Any]:
    meta = get_file_metadata(path)
    return {
        "filename": pub["filename"],
        "source_url": pub["url"],
        "domain": pub["domain"],
        "subdomain": "nist_publications",
        "doc_type": pub["doc_type"],
        "pages": meta["pages"],
        "download_date": datetime.utcnow().date().isoformat(),
        "size_kb": meta["size_kb"],
        "description": pub["description"],
    }
