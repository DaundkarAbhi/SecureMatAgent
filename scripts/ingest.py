#!/usr/bin/env python
"""
CLI entrypoint for the SecureMatAgent ingestion pipeline.

Usage:
    python scripts/ingest.py --data-dir ./data/documents
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src` and `config` are importable
# when this script is run directly (e.g. python scripts/ingest.py).
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SecureMatAgent document ingestion pipeline."
    )
    parser.add_argument(
        "--data-dir",
        default="./data/documents",
        help="Directory containing source documents (PDF, TXT, MD). "
        "Default: ./data/documents",
    )
    return parser.parse_args(argv)


def _print_summary(result: dict) -> None:
    """Print a formatted summary table to stdout."""
    bar = "=" * 52
    print()
    print(bar)
    print("  SecureMatAgent - Ingestion Summary")
    print(bar)
    print(f"  {'Source files processed:':<32} {result['total_files']:>6}")
    print(f"  {'Chunks created:':<32} {result['total_chunks']:>6}")
    print(f"  {'Total vectors in collection:':<32} {result['collection_size']:>6}")
    print()
    print("  Domain breakdown (chunks):")
    domain_breakdown = result.get("domain_breakdown", {})
    if domain_breakdown:
        for domain, count in sorted(domain_breakdown.items()):
            print(f"    {domain:<28} {count:>6}")
    else:
        print("    (none)")
    print(bar)
    print()


def main(argv: list = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    _configure_logging()
    args = _parse_args(argv)

    logger = logging.getLogger(__name__)
    logger.info("Starting ingestion pipeline. data_dir='%s'", args.data_dir)

    try:
        from src.ingestion.pipeline import run_ingestion

        result = run_ingestion(data_dir=args.data_dir)
        _print_summary(result)
        return 0
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user.")
        return 1
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
