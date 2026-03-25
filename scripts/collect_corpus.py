#!/usr/bin/env python3
"""
SecureMatAgent Corpus Collector — main orchestrator.

Usage:
    python scripts/collect_corpus.py --output-dir ./data/documents --verbose

Calls each collector module, builds a manifest JSON, and prints a summary.
Idempotent: already-downloaded files are skipped.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure the scripts/ directory is on the path so collectors can be imported
sys.path.insert(0, str(Path(__file__).parent))

from collectors import (arxiv_collector, custom_docs, msds_collector,
                        nist_collector)

# ── Logging setup ──────────────────────────────────────────────────────────


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "requests", "arxiv", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Manifest helpers ───────────────────────────────────────────────────────


def load_manifest(path: Path) -> dict:
    """Load existing manifest or return empty structure."""
    if path.exists():
        try:
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            logging.warning("Corrupt manifest at %s — starting fresh", path)
    return {"generated_at": "", "documents": []}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["generated_at"] = datetime.utcnow().isoformat() + "Z"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def merge_entries(existing: list[dict], new_entries: list[dict]) -> list[dict]:
    """
    Merge *new_entries* into *existing*, replacing by filename (idempotent).
    """
    index = {e["filename"]: e for e in existing}
    for entry in new_entries:
        index[entry["filename"]] = entry
    return list(index.values())


# ── Collection pipeline ────────────────────────────────────────────────────

COLLECTORS = [
    ("ArXiv papers", arxiv_collector),
    ("NIST publications", nist_collector),
    ("MSDS/SDS documents", msds_collector),
    ("Custom lab documents", custom_docs),
]


def run_collection(output_dir: Path, manifest_path: Path, verbose: bool) -> dict:
    manifest = load_manifest(manifest_path)
    all_entries: list[dict] = manifest.get("documents", [])
    collector_results: dict[str, int] = {}
    failed_collectors: list[str] = []

    for label, module in COLLECTORS:
        print(f"\n{'─' * 60}")
        print(f"  Running: {label}")
        print(f"{'─' * 60}")

        start_time = time.monotonic()
        try:
            entries = module.collect(output_dir=output_dir, verbose=verbose)
            elapsed = time.monotonic() - start_time
            all_entries = merge_entries(all_entries, entries)
            collector_results[label] = len(entries)
            print(f"  ✓ {label}: {len(entries)} document(s) in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logging.error("Collector '%s' failed after %.1fs: %s", label, elapsed, exc)
            failed_collectors.append(label)
            collector_results[label] = 0
            print(f"  ✗ {label}: FAILED — {exc}")

        # Save progress after each collector (crash-safe)
        manifest["documents"] = all_entries
        save_manifest(manifest_path, manifest)

    manifest["collector_results"] = collector_results
    manifest["failed_collectors"] = failed_collectors
    manifest["documents"] = all_entries
    save_manifest(manifest_path, manifest)

    return manifest


# ── Summary report ─────────────────────────────────────────────────────────


def print_summary(manifest: dict, output_dir: Path) -> None:
    docs = manifest.get("documents", [])
    if not docs:
        print("\nNo documents collected.")
        return

    # Domain breakdown
    domain_counts: dict[str, int] = {}
    domain_sizes: dict[str, float] = {}
    total_size = 0.0
    missing: list[str] = []

    for doc in docs:
        domain = doc.get("domain", "unknown")
        size = doc.get("size_kb", 0)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        domain_sizes[domain] = domain_sizes.get(domain, 0) + size
        total_size += size

        if not (output_dir / doc["filename"]).exists():
            missing.append(doc["filename"])

    # ── Header ────────────────────────────────────────────────────────────
    width = 68
    print("\n" + "═" * width)
    print("  CORPUS COLLECTION SUMMARY")
    print("═" * width)
    print(f"  Total documents : {len(docs)}")
    print(f"  Total size      : {total_size / 1024:.2f} MB")
    print(f"  Output dir      : {output_dir}")
    print(f"  Manifest        : {output_dir.parent / 'corpus_manifest.json'}")

    # ── Per-domain breakdown ───────────────────────────────────────────────
    print(f"\n  {'Domain':<30} {'Docs':>5}  {'Size':>9}")
    print("  " + "─" * (width - 2))
    for domain in sorted(domain_counts):
        count = domain_counts[domain]
        size_mb = domain_sizes[domain] / 1024
        print(f"  {domain:<30} {count:>5}  {size_mb:>7.2f} MB")

    # ── Per-collector breakdown ────────────────────────────────────────────
    if "collector_results" in manifest:
        print(f"\n  {'Collector':<35} {'Docs':>5}")
        print("  " + "─" * (width - 2))
        for col, cnt in manifest["collector_results"].items():
            print(f"  {col:<35} {cnt:>5}")

    # ── Failures ──────────────────────────────────────────────────────────
    if manifest.get("failed_collectors"):
        print(f"\n  ⚠  Failed collectors: {', '.join(manifest['failed_collectors'])}")
        print("     Re-run the script to retry failed downloads.")

    if missing:
        print(
            f"\n  ⚠  {len(missing)} file(s) listed in manifest but not found on disk:"
        )
        for f in missing:
            print(f"     — {f}")

    print("\n" + "═" * width)

    # ── File listing ──────────────────────────────────────────────────────
    print(f"\n  {'Filename':<55} {'KB':>7}  {'Pages':>5}  {'Domain'}")
    print("  " + "─" * (width + 10))
    for doc in sorted(docs, key=lambda d: d.get("domain", "")):
        fname = doc["filename"]
        kb = doc.get("size_kb", 0)
        pages = doc.get("pages", "-")
        domain = doc.get("domain", "")
        exists_mark = "" if (output_dir / fname).exists() else "  [MISSING]"
        pages_str = str(pages) if pages else "—"
        print(f"  {fname:<55} {kb:>7.0f}  {pages_str:>5}  {domain}{exists_mark}")

    print()


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SecureMatAgent document corpus collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/collect_corpus.py
  python scripts/collect_corpus.py --output-dir ./data/documents --verbose
  python scripts/collect_corpus.py --skip arxiv nist
""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/documents"),
        help="Directory to save documents (default: ./data/documents)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest JSON (default: <output-dir>/../corpus_manifest.json)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["arxiv", "nist", "msds", "custom"],
        default=[],
        help="Skip specific collectors",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Ensure UTF-8 output on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    setup_logging(args.verbose)

    output_dir: Path = args.output_dir.resolve()
    manifest_path: Path = args.manifest or (output_dir.parent / "corpus_manifest.json")

    # Apply --skip filter
    global COLLECTORS
    skip_map = {
        "arxiv": "ArXiv papers",
        "nist": "NIST publications",
        "msds": "MSDS/SDS documents",
        "custom": "Custom lab documents",
    }
    if args.skip:
        skip_labels = {skip_map[s] for s in args.skip}
        COLLECTORS = [(lbl, mod) for lbl, mod in COLLECTORS if lbl not in skip_labels]
        print(f"Skipping: {', '.join(args.skip)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSecureMatAgent Corpus Collector")
    print(f"Output directory : {output_dir}")
    print(f"Manifest         : {manifest_path}")
    print(f"Collectors       : {', '.join(lbl for lbl, _ in COLLECTORS)}")

    manifest = run_collection(output_dir, manifest_path, args.verbose)
    print_summary(manifest, output_dir)

    failed = manifest.get("failed_collectors", [])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
