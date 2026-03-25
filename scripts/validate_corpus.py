#!/usr/bin/env python3
"""
SecureMatAgent Corpus Validator

Validates the collected document corpus:
  - All files readable and non-corrupt
  - No PDFs smaller than 1 KB
  - Domain distribution looks reasonable
  - Manifest is consistent with files on disk

Usage:
    python scripts/validate_corpus.py
    python scripts/validate_corpus.py --output-dir ./data/documents --verbose
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure scripts/ is on path
sys.path.insert(0, str(Path(__file__).parent))


# ── Validation helpers ─────────────────────────────────────────────────────


def load_manifest(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [WARN] Manifest not found: {path}")
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"  [ERROR] Cannot parse manifest: {exc}")
        return None


def validate_pdf(path: Path) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    if not path.exists():
        return False, "file not found"
    size = path.stat().st_size
    if size < 512:
        return False, f"too small ({size} bytes)"

    # Check PDF magic bytes
    try:
        with path.open("rb") as fh:
            header = fh.read(8)
        if not header.startswith(b"%PDF"):
            return False, f"not a PDF (header: {header[:8]!r})"
    except OSError as exc:
        return False, f"cannot read file: {exc}"

    # Try to parse with pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        if page_count == 0:
            return False, "PDF has 0 pages"
        return True, f"OK ({page_count} pages)"
    except ImportError:
        return True, "OK (pypdf not installed, magic bytes only)"
    except Exception as exc:
        return False, f"pypdf error: {exc}"


def validate_text(path: Path) -> tuple[bool, str]:
    """Return (is_valid, reason) for text/markdown files."""
    if not path.exists():
        return False, "file not found"
    size = path.stat().st_size
    if size < 100:
        return False, f"suspiciously small ({size} bytes)"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        words = len(text.split())
        return True, f"OK ({words} words, {size // 1024} KB)"
    except Exception as exc:
        return False, f"cannot read: {exc}"


def validate_file(path: Path) -> tuple[bool, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return validate_pdf(path)
    elif suffix in (".md", ".txt", ".json"):
        return validate_text(path)
    else:
        return path.exists(), "unknown type — existence check only"


# ── Main validation logic ──────────────────────────────────────────────────

MIN_FILE_SIZE_KB = 1.0
EXPECTED_DOMAINS = {"materials_science", "cybersecurity"}
MIN_DOCS_PER_DOMAIN = 3
TARGET_TOTAL_DOCS = 25


def run_validation(output_dir: Path, manifest_path: Path, verbose: bool) -> int:
    """Returns number of validation errors found."""
    errors = 0
    warnings = 0

    print(f"\nSecureMatAgent Corpus Validator")
    print(f"Output dir : {output_dir}")
    print(f"Manifest   : {manifest_path}\n")

    # ── Load manifest ──────────────────────────────────────────────────────
    manifest = load_manifest(manifest_path)
    if manifest is None:
        print("[ERROR] Cannot load manifest — aborting validation.")
        return 1

    docs = manifest.get("documents", [])
    print(f"Manifest contains {len(docs)} document entries.\n")

    # ── Scan actual files on disk ──────────────────────────────────────────
    all_files = list(output_dir.glob("*"))
    doc_files = [
        f
        for f in all_files
        if f.suffix.lower() in (".pdf", ".md", ".txt", ".json")
        and not f.name.startswith(".")
    ]

    # ── Validate each manifest entry ───────────────────────────────────────
    width = 70
    print(f"  {'Filename':<50} {'Size KB':>8}  {'Status'}")
    print("  " + "─" * width)

    domain_counts: dict[str, int] = {}
    manifested_files: set[str] = set()

    for doc in sorted(docs, key=lambda d: (d.get("domain", ""), d["filename"])):
        fname = doc["filename"]
        fpath = output_dir / fname
        manifested_files.add(fname)
        domain = doc.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        size_kb = fpath.stat().st_size / 1024 if fpath.exists() else 0

        is_valid, reason = validate_file(fpath)
        status_icon = "✓" if is_valid else "✗"

        if not is_valid:
            errors += 1
            status_str = f"[ERROR] {reason}"
        elif size_kb < MIN_FILE_SIZE_KB:
            warnings += 1
            status_str = f"[WARN ] {reason} — tiny file"
        else:
            status_str = f"[OK   ] {reason}"

        print(f"  {fname:<50} {size_kb:>8.1f}  {status_icon} {status_str}")

    # ── Check for disk files not in manifest ──────────────────────────────
    unmanifested = [f.name for f in doc_files if f.name not in manifested_files]
    if unmanifested:
        print(f"\n  [INFO] Files on disk not in manifest ({len(unmanifested)}):")
        for fn in sorted(unmanifested):
            sz = (output_dir / fn).stat().st_size / 1024
            print(f"    {fn}  ({sz:.1f} KB)")

    # ── Domain distribution check ──────────────────────────────────────────
    print(f"\n  {'Domain Distribution':}")
    print("  " + "─" * 40)
    total_docs = sum(domain_counts.values())
    for domain in sorted(domain_counts):
        count = domain_counts[domain]
        bar = "█" * count
        print(f"  {domain:<28} {count:>3}  {bar}")

    for expected_domain in EXPECTED_DOMAINS:
        count = domain_counts.get(expected_domain, 0)
        if count < MIN_DOCS_PER_DOMAIN:
            warnings += 1
            print(
                f"  [WARN] Domain '{expected_domain}' has only {count} doc(s) "
                f"(expected ≥ {MIN_DOCS_PER_DOMAIN})"
            )

    # ── Total count check ──────────────────────────────────────────────────
    print(f"\n  Total documents: {total_docs}")
    if total_docs < TARGET_TOTAL_DOCS:
        warnings += 1
        print(
            f"  [WARN] Only {total_docs} documents collected; "
            f"target is {TARGET_TOTAL_DOCS}+. "
            "Re-run collect_corpus.py to fill gaps."
        )
    else:
        print(f"  [OK  ] Document count meets target (≥{TARGET_TOTAL_DOCS})")

    # ── Total size ─────────────────────────────────────────────────────────
    total_kb = sum(
        (output_dir / doc["filename"]).stat().st_size / 1024
        for doc in docs
        if (output_dir / doc["filename"]).exists()
    )
    print(f"  Total size: {total_kb / 1024:.2f} MB")

    # ── Failed collectors ──────────────────────────────────────────────────
    failed = manifest.get("failed_collectors", [])
    if failed:
        warnings += 1
        print(f"\n  [WARN] Collectors that failed: {', '.join(failed)}")

    # ── Final verdict ──────────────────────────────────────────────────────
    print(f"\n{'═' * (width + 4)}")
    if errors == 0 and warnings == 0:
        print(f"  PASS — {total_docs} documents, 0 errors, 0 warnings")
    elif errors == 0:
        print(
            f"  PASS with warnings — {total_docs} docs, 0 errors, {warnings} warning(s)"
        )
    else:
        print(f"  FAIL — {errors} error(s), {warnings} warning(s)")
    print(f"{'═' * (width + 4)}\n")

    return errors


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the SecureMatAgent document corpus"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/documents"),
        help="Directory containing collected documents (default: ./data/documents)",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Ensure UTF-8 output on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest or (output_dir.parent / "corpus_manifest.json")
    errors = run_validation(output_dir, manifest_path, args.verbose)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
