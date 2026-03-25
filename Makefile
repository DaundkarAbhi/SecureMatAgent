# ============================================================
#  SecureMatAgent Makefile
#  Usage: make <target>
# ============================================================

SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

VENV := .venv
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip
COMPOSE := docker compose

# Colours
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RESET  := \033[0m

.PHONY: help build up down logs ingest test test-unit test-cov shell clean check api ui eval eval-quick eval-charts

## help          → Show this help message
help:
	@echo ""
	@echo "  SecureMatAgent — available targets"
	@echo "  ─────────────────────────────────────────────────────"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  make /'
	@echo ""

## api           → Run FastAPI dev server on port 8000 (local venv)
api: _require_venv
	@echo -e "$(GREEN)Starting FastAPI on http://localhost:8000/docs …$(RESET)"
	LOCAL_DEV=true $(VENV)/Scripts/uvicorn src.api.main:app --reload --port 8000 --host 0.0.0.0

## ui            → Run Streamlit UI on port 8501 (local venv)
ui: _require_venv
	@echo -e "$(GREEN)Starting Streamlit on http://localhost:8501 …$(RESET)"
	LOCAL_DEV=true $(VENV)/Scripts/streamlit run src/frontend/app.py --server.port 8501

## build         → Build the Docker image
build:
	@echo -e "$(GREEN)Building Docker image…$(RESET)"
	$(COMPOSE) build --no-cache

## up            → Start Docker services (app + qdrant) after verifying Ollama
up: _check_ollama
	@echo -e "$(GREEN)Starting services…$(RESET)"
	$(COMPOSE) up -d
	@echo -e "$(GREEN)Services started. Endpoints:$(RESET)"
	@echo "  FastAPI  : http://localhost:8000/docs"
	@echo "  Streamlit: http://localhost:8501"
	@echo "  Qdrant   : http://localhost:6333/dashboard"

## down          → Stop and remove Docker containers
down:
	@echo -e "$(YELLOW)Stopping services…$(RESET)"
	$(COMPOSE) down

## logs          → Tail logs from all containers
logs:
	$(COMPOSE) logs -f --tail=100

## ingest        → Run the document ingestion pipeline (in local venv)
ingest: _require_venv
	@echo -e "$(GREEN)Running ingestion pipeline…$(RESET)"
	LOCAL_DEV=true $(PYTHON) -m app.ingestion.ingest

## test          → Run unit + integration tests (pytest, in local venv)
test: _require_venv
	@echo -e "$(GREEN)Running tests…$(RESET)"
	LOCAL_DEV=true $(PYTHON) -m pytest tests/ -v --tb=short

## test-unit     → Run only unit tests
test-unit: _require_venv
	@echo -e "$(GREEN)Running unit tests…$(RESET)"
	LOCAL_DEV=true $(PYTHON) -m pytest tests/unit/ -v --tb=short -m unit

## test-cov      → Run tests with coverage report
test-cov: _require_venv
	@echo -e "$(GREEN)Running tests with coverage…$(RESET)"
	LOCAL_DEV=true $(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=src --cov=config \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

## eval          → Run full evaluation (RAGAS + tool accuracy, 30 questions)
eval: _require_venv
	@echo -e "$(GREEN)Running full evaluation pipeline…$(RESET)"
	@echo -e "$(YELLOW)  Requires: Ollama running + Qdrant up + eval deps installed$(RESET)"
	@echo -e "$(YELLOW)  Install eval deps: $(PIP) install -r eval/requirements-eval.txt$(RESET)"
	LOCAL_DEV=true $(PYTHON) eval/run_eval.py \
		--test-set eval/test_set.json \
		--output eval/results/ \
		--visualize

## eval-quick    → Run quick evaluation (10 questions, no RAGAS, ~5× faster)
eval-quick: _require_venv
	@echo -e "$(GREEN)Running quick evaluation (10 questions, custom metrics only)…$(RESET)"
	LOCAL_DEV=true $(PYTHON) eval/run_eval.py \
		--test-set eval/test_set.json \
		--output eval/results/ \
		--quick \
		--no-ragas

## eval-charts   → Regenerate charts from existing eval/results/ JSON files
eval-charts: _require_venv
	@echo -e "$(GREEN)Generating evaluation charts…$(RESET)"
	LOCAL_DEV=true $(PYTHON) eval/visualize.py \
		--ragas-json eval/results/ragas_scores.json \
		--tool-json  eval/results/tool_accuracy.json \
		--output     eval/results/charts/

## check         → Verify Ollama + Qdrant are healthy (local)
check: _require_venv
	LOCAL_DEV=true $(PYTHON) scripts/check_services.py

## shell         → Open a shell inside the running app container
shell:
	$(COMPOSE) exec app /bin/bash

## clean         → Remove containers, volumes, __pycache__, and .pytest_cache
clean:
	@echo -e "$(YELLOW)Cleaning up…$(RESET)"
	$(COMPOSE) down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done."

# ------------------------------------------------------------------ #
# Internal helpers (not listed in help)
# ------------------------------------------------------------------ #

_check_ollama:
	@echo -e "$(YELLOW)Checking native Ollama…$(RESET)"
	@if ! command -v ollama &>/dev/null; then \
		echo "WARNING: 'ollama' not found in PATH. Is it installed?"; \
	else \
		ollama list || echo "WARNING: Could not list Ollama models (is 'ollama serve' running?)"; \
	fi

_require_venv:
	@if [ ! -f "$(PYTHON)" ]; then \
		echo "ERROR: venv not found at $(VENV). Create it with:"; \
		echo "  python -m venv $(VENV) && $(PIP) install -r requirements.txt"; \
		exit 1; \
	fi
