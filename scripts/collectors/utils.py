"""
Shared utilities for corpus collection.
Provides: download_pdf, extract_pdf_pages, validate_pdf,
          sanitize_filename, get_file_metadata, retry logic.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── HTTP session shared across all collectors ──────────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "SecureMatAgent-CorpusCollector/1.0 "
            "(academic research tool; contact: research@example.edu)"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*",
    }
)


# ── Retry decorator ────────────────────────────────────────────────────────


def with_retry(max_attempts: int = 3, base_delay: float = 2.0):
    """Exponential-backoff retry decorator."""
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            fn.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "%s attempt %d/%d failed (%s) — retrying in %.1fs",
                        fn.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


# ── Core helpers ───────────────────────────────────────────────────────────


@with_retry(max_attempts=3, base_delay=2.0)
def download_pdf(url: str, output_path: str | Path, timeout: int = 60) -> bool:
    """
    Download a file from *url* to *output_path*.

    Returns True on success, False on failure (never raises so callers can
    decide whether to continue).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading  %s", url)
    response = _SESSION.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type and "pdf" not in url.lower():
        logger.warning("Response looks like HTML, not a PDF — skipping %s", url)
        return False

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with output_path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)

    size_kb = output_path.stat().st_size / 1024
    logger.info("  → saved %.1f KB to %s", size_kb, output_path.name)

    if size_kb < 1:
        logger.warning("  ! file is suspiciously small (%.1f KB)", size_kb)
        return False

    return True


def extract_pdf_pages(
    input_path: str | Path,
    output_path: str | Path,
    start_page: int,
    end_page: int,
) -> bool:
    """
    Extract pages [start_page, end_page] (1-based, inclusive) from *input_path*
    and write them to *output_path*.

    Returns True on success.
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter  # type: ignore
        except ImportError:
            logger.error("Neither pypdf nor PyPDF2 is installed — cannot extract pages")
            return False

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Some complex NIST PDFs hit Python's default recursion limit
        import sys as _sys

        old_limit = _sys.getrecursionlimit()
        _sys.setrecursionlimit(max(old_limit, 10000))
        reader = PdfReader(str(input_path))
        total_pages = len(reader.pages)
        _sys.setrecursionlimit(old_limit)
        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, end_page)

        if start_idx >= total_pages:
            logger.warning(
                "start_page=%d exceeds total pages=%d for %s",
                start_page,
                total_pages,
                input_path.name,
            )
            # Fall back to first 40 pages
            start_idx = 0
            end_idx = min(40, total_pages)

        writer = PdfWriter()
        for i in range(start_idx, end_idx):
            writer.add_page(reader.pages[i])

        with output_path.open("wb") as fh:
            writer.write(fh)

        extracted = end_idx - start_idx
        logger.info(
            "  → extracted pages %d–%d (%d pages) → %s",
            start_idx + 1,
            end_idx,
            extracted,
            output_path.name,
        )
        return True

    except Exception as exc:
        logger.error("Failed to extract pages from %s: %s", input_path.name, exc)
        return False


def validate_pdf(path: str | Path) -> bool:
    """Return True if *path* is a readable, non-empty PDF."""
    path = Path(path)
    if not path.exists() or path.stat().st_size < 512:
        return False
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            # Can't validate without library — trust the file
            return path.stat().st_size > 512

    try:
        reader = PdfReader(str(path))
        return len(reader.pages) > 0
    except Exception:
        return False


def sanitize_filename(text: str) -> str:
    """
    Convert *text* to a safe filename component.
    Replaces spaces with underscores, strips special chars.
    """
    text = text.strip()
    text = re.sub(r"[^\w\s\-.]", "", text)
    text = re.sub(r"[\s]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text[:80]  # cap length


def get_file_metadata(path: str | Path) -> dict:
    """Return size_kb, pages, type for *path*."""
    path = Path(path)
    if not path.exists():
        return {"size_kb": 0, "pages": 0, "type": "missing"}

    suffix = path.suffix.lower()
    size_kb = round(path.stat().st_size / 1024, 1)

    pages = 0
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except ImportError:
                PdfReader = None  # type: ignore

        if PdfReader is not None:
            try:
                pages = len(PdfReader(str(path)).pages)
            except Exception:
                pages = 0

    doc_type = {
        ".pdf": "PDF",
        ".md": "Markdown",
        ".txt": "Text",
        ".json": "JSON",
    }.get(suffix, suffix.lstrip(".").upper())

    return {"size_kb": size_kb, "pages": pages, "type": doc_type}
