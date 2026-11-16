.PHONY: help setup data dbt baselines als ranker rolling suitability evaluate test lint format all

help:  ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

data:  ## Download FAR-Trans into data/raw
	uv run python scripts/download_data.py

dbt:  ## Build the DuckDB warehouse (staging + interactions + features)
	cd dbt && uv run dbt build --profiles-dir . --project-dir .

baselines:  ## Evaluate popularity / recent-popularity / item-item baselines
	uv run python -m recs.pipelines.baselines

als:  ## Tune ALS and BPR on validation and log to MLflow
	uv run python -m recs.pipelines.train_mf

ranker:  ## Train the LightGBM LambdaRank ranker on top-50 candidates
	uv run python -m recs.pipelines.train_ranker

rolling:  ## Ranker + baselines on the FAR-Trans rolling benchmark (31 cutoffs, about 1 hour)
	uv run python -m recs.pipelines.train_ranker --rolling-step 2

suitability:  ## Measure the NDCG cost of the suitability rules
	uv run python -m recs.pipelines.suitability_report

evaluate: baselines als ranker suitability  ## Run every offline evaluation

test:  ## Run the test suite
	uv run pytest

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .

all: data dbt evaluate  ## Download, build the warehouse, run the evaluation
