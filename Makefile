.PHONY: help setup data warehouse baselines evaluate test lint format all

help:  ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

data:  ## Download FAR-Trans into data/raw
	uv run python scripts/download_data.py

warehouse:  ## Load the csvs into DuckDB
	uv run python -m recs.warehouse

baselines:  ## Evaluate popularity / recent-popularity / item-item baselines
	uv run python -m recs.pipelines.baselines

evaluate: baselines  ## Run the offline evaluation

test:  ## Run the test suite
	uv run pytest

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .

all: data warehouse evaluate  ## Download, build the warehouse, run the evaluation
