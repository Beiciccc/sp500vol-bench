.PHONY: help setup test lint format spike data data-sample train eval ablation tables figures paper dissertation clean

SHELL := /bin/bash
PYTHON := uv run python

help:
	@echo "SP500Vol-Bench — common tasks"
	@echo ""
	@echo "  setup           Install deps via uv"
	@echo "  test            Run pytest"
	@echo "  lint            ruff + mypy"
	@echo "  format          ruff format"
	@echo ""
	@echo "  spike           Prototype spike: 5 firms × 1 year, end-to-end"
	@echo "  data-sample     Generate small sample dataset for tests"
	@echo "  data            Full S&P 500 ingestion (EDGAR + market + alignment + labels)"
	@echo ""
	@echo "  train MODEL=ID  Train model (e.g. MODEL=C2_finbert_s3)"
	@echo "  eval RUN=ID     Evaluate a finished run"
	@echo "  ablation AB=ID  Run an ablation (e.g. AB=AB1)"
	@echo ""
	@echo "  tables          Regenerate result tables"
	@echo "  figures         Regenerate figures"
	@echo "  paper           Compile the two-column manuscript PDF"
	@echo "  dissertation    Compile MSc dissertation PDF"
	@echo ""
	@echo "  clean           Remove build artefacts"

# === Setup ===
setup:
	uv sync --extra dev --extra viz

# === Quality gates ===
test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -m "not slow and not gpu and not network"

lint:
	uv run ruff check src/ tests/ scripts/
	uv run mypy src/sp500vol/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

# === Data ===
spike:
	$(PYTHON) scripts/run_spike.py

data-sample:
	$(PYTHON) scripts/build_dataset.py --config configs/data/sample.yaml

data:
	$(PYTHON) scripts/build_dataset.py --config configs/data/full.yaml

# === Experiments ===
train:
	@if [ -z "$(MODEL)" ]; then echo "Usage: make train MODEL=C2_finbert_s3"; exit 1; fi
	$(PYTHON) scripts/train.py --model $(MODEL)

eval:
	@if [ -z "$(RUN)" ]; then echo "Usage: make eval RUN=<run_id>"; exit 1; fi
	$(PYTHON) scripts/evaluate.py --run-id $(RUN)

ablation:
	@if [ -z "$(AB)" ]; then echo "Usage: make ablation AB=AB1"; exit 1; fi
	$(PYTHON) scripts/run_ablation.py --ablation $(AB)

# === Reporting ===
tables:
	$(PYTHON) scripts/analysis/aggregate_seeds.py
	$(PYTHON) scripts/analysis/check_convergence.py

figures:
	@echo "figure generation: add scripts/make_figures.py once 4xA100 results land"

paper:
	cd writing/paper && latexmk -pdf main.tex

dissertation:
	cd writing/dissertation && latexmk -pdf main.tex

# === Cleanup ===
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
