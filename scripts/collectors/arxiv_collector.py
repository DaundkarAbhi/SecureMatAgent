"""
ArXiv collector — downloads materials-science PDFs from arxiv.org.

Covers:
  - Perovskite crystal structure / halide perovskite solar cells
  - Lithium-ion cathode / solid-state electrolyte
  - XRD / Rietveld refinement / powder diffraction
  - High-entropy ceramics / thermal barrier coatings

Target: 12-14 papers total.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Search configurations ──────────────────────────────────────────────────
SEARCH_CONFIGS = [
    # (domain_tag, query, max_results, category_label)
    ("perovskite", "perovskite crystal structure review", 2, "materials_science"),
    ("perovskite", "halide perovskite solar cell efficiency", 2, "materials_science"),
    ("battery", "lithium ion cathode materials capacity", 2, "materials_science"),
    ("battery", "solid state electrolyte lithium", 2, "materials_science"),
    ("xrd", "X-ray diffraction Rietveld refinement powder", 2, "crystallography"),
    ("xrd", "powder diffraction peak analysis crystal structure", 1, "crystallography"),
    ("ceramics", "high entropy ceramics mechanical properties", 2, "materials_science"),
    ("ceramics", "thermal barrier coating zirconia", 1, "materials_science"),
]


def _make_filename(domain_tag: str, paper, index: int) -> str:
    """Build a descriptive filename from arxiv metadata."""
    from .utils import sanitize_filename

    # Shorten title to first 6 words
    title_words = paper.title.split()[:6]
    short_title = "_".join(title_words)
    short_title = sanitize_filename(short_title)

    year = paper.published.year if paper.published else "unknown"
    arxiv_id_short = paper.get_short_id().replace("/", "_").replace(".", "_")

    return f"arxiv_{domain_tag}_{short_title}_{year}_{arxiv_id_short}.pdf"


def collect(output_dir: Path, verbose: bool = False) -> list[dict[str, Any]]:
    """
    Run all searches, download PDFs, return list of manifest entries.
    Idempotent — skips files that already exist.
    """
    try:
        import arxiv  # type: ignore
    except ImportError:
        logger.error("'arxiv' package not installed. Run: pip install arxiv")
        return []

    from .utils import get_file_metadata, validate_pdf

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for domain_tag, query, max_results, category_label in SEARCH_CONFIGS:
        logger.info("Searching arXiv: %r (max %d)", query, max_results)
        time.sleep(1)  # rate-limit between searches

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = list(search.results())
        except Exception as exc:
            logger.warning("Search failed for %r: %s — skipping", query, exc)
            continue

        for idx, paper in enumerate(results):
            arxiv_id = paper.get_short_id()

            if arxiv_id in seen_ids:
                logger.debug("Duplicate paper %s — skipping", arxiv_id)
                continue
            seen_ids.add(arxiv_id)

            filename = _make_filename(domain_tag, paper, idx)
            out_path = output_dir / filename

            if out_path.exists() and validate_pdf(out_path):
                logger.info("  [skip] %s already exists", filename)
            else:
                logger.info("  Downloading %s  (%s)", arxiv_id, paper.title[:60])
                try:
                    # arxiv library handles the PDF download
                    paper.download_pdf(dirpath=str(output_dir), filename=filename)
                    time.sleep(1.5)  # polite rate-limit
                except Exception as exc:
                    logger.warning("  Failed to download %s: %s", arxiv_id, exc)
                    continue

            if not out_path.exists():
                logger.warning("  PDF not found after download attempt: %s", filename)
                continue

            meta = get_file_metadata(out_path)
            authors_str = ", ".join(str(a) for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."

            manifest_entries.append(
                {
                    "filename": filename,
                    "source_url": str(paper.pdf_url),
                    "domain": "materials_science",
                    "subdomain": domain_tag,
                    "doc_type": "research_paper",
                    "pages": meta["pages"],
                    "download_date": datetime.utcnow().date().isoformat(),
                    "size_kb": meta["size_kb"],
                    "description": paper.title,
                    "arxiv_id": arxiv_id,
                    "authors": authors_str,
                    "abstract": (
                        paper.summary[:300] + "..."
                        if len(paper.summary) > 300
                        else paper.summary
                    ),
                    "published_date": (
                        paper.published.date().isoformat()
                        if paper.published
                        else "unknown"
                    ),
                    "category": category_label,
                }
            )

            if verbose:
                print(
                    f"    ✓ {filename}  ({meta['size_kb']:.0f} KB, {meta['pages']} pages)"
                )

    logger.info("ArXiv collector finished: %d papers collected", len(manifest_entries))
    return manifest_entries
