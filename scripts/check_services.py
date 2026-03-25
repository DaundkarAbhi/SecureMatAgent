"""
SecureMatAgent — Service health check script.

Verifies:
  1. Ollama is running at localhost:11434 and the 'mistral' model is available.
  2. Qdrant is running at localhost:6333.

Usage:
    python scripts/check_services.py
    python scripts/check_services.py --ollama-url http://localhost:11434 --qdrant-url http://localhost:6333
"""

from __future__ import annotations

import argparse
import sys

import httpx

# ------------------------------------------------------------------ #
# ANSI colours
# ------------------------------------------------------------------ #
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _ok(msg: str) -> str:
    return f"{GREEN}✔  {msg}{RESET}"


def _fail(msg: str) -> str:
    return f"{RED}✘  {msg}{RESET}"


def _warn(msg: str) -> str:
    return f"{YELLOW}⚠  {msg}{RESET}"


# ------------------------------------------------------------------ #
# Checks
# ------------------------------------------------------------------ #


def check_ollama(base_url: str, expected_model: str = "mistral") -> bool:
    print(f"\n{BOLD}[Ollama]{RESET}  {base_url}")
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        resp = httpx.get(url, timeout=5)
        resp.raise_for_status()
    except httpx.ConnectError:
        print(_fail(f"Cannot connect to Ollama at {base_url}"))
        print("        Is Ollama running?  Try: ollama serve")
        return False
    except httpx.HTTPStatusError as exc:
        print(_fail(f"HTTP {exc.response.status_code} from {url}"))
        return False
    except Exception as exc:
        print(_fail(f"Unexpected error: {exc}"))
        return False

    print(_ok("Ollama is reachable"))

    data = resp.json()
    models = [m.get("name", "") for m in data.get("models", [])]

    if not models:
        print(_warn("No models pulled yet. Run: ollama pull mistral"))
        return False

    # Check if expected model is present (partial match e.g. "mistral:latest")
    matched = [m for m in models if expected_model in m]
    if matched:
        print(_ok(f"Model '{expected_model}' found: {matched[0]}"))
        print(f"        All models: {models}")
        return True
    else:
        print(_warn(f"Model '{expected_model}' NOT found. Available: {models}"))
        print(f"        Run: ollama pull {expected_model}")
        return False


def check_qdrant(base_url: str) -> bool:
    print(f"\n{BOLD}[Qdrant]{RESET}  {base_url}")
    healthz_url = f"{base_url.rstrip('/')}/healthz"
    collections_url = f"{base_url.rstrip('/')}/collections"
    try:
        resp = httpx.get(healthz_url, timeout=5)
        resp.raise_for_status()
    except httpx.ConnectError:
        print(_fail(f"Cannot connect to Qdrant at {base_url}"))
        print("        Is Qdrant running?  Try: docker compose up qdrant -d")
        return False
    except httpx.HTTPStatusError as exc:
        print(_fail(f"HTTP {exc.response.status_code} from {healthz_url}"))
        return False
    except Exception as exc:
        print(_fail(f"Unexpected error: {exc}"))
        return False

    print(_ok("Qdrant is reachable and healthy"))

    # Bonus: list collections
    try:
        r2 = httpx.get(collections_url, timeout=5)
        r2.raise_for_status()
        cols = [c["name"] for c in r2.json().get("result", {}).get("collections", [])]
        if cols:
            print(f"        Collections: {cols}")
        else:
            print("        No collections yet (run ingest first).")
    except Exception:
        pass  # Not critical

    return True


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main() -> int:
    parser = argparse.ArgumentParser(description="SecureMatAgent service health check")
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="Ollama base URL"
    )
    parser.add_argument(
        "--qdrant-url", default="http://localhost:6333", help="Qdrant base URL"
    )
    parser.add_argument("--model", default="mistral", help="Expected Ollama model name")
    args = parser.parse_args()

    print(f"\n{BOLD}SecureMatAgent — Service Health Check{RESET}")
    print("=" * 45)

    ollama_ok = check_ollama(args.ollama_url, args.model)
    qdrant_ok = check_qdrant(args.qdrant_url)

    print("\n" + "=" * 45)
    if ollama_ok and qdrant_ok:
        print(_ok("All services are UP. SecureMatAgent is ready to run."))
        return 0
    else:
        failing = []
        if not ollama_ok:
            failing.append("Ollama")
        if not qdrant_ok:
            failing.append("Qdrant")
        print(_fail(f"Some services are DOWN: {', '.join(failing)}"))
        print("       Fix the issues above, then re-run this script.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
