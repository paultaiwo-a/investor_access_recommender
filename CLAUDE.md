# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## What this is
A two-stage financial asset recommender on the FAR-Trans dataset (National Bank of Greece
retail transactions, 2018-2022). Candidates come from ALS / BPR / item-item models, a LightGBM
LambdaRank model reorders them, and a suitability layer enforces risk and concentration rules.

## Layout
- `data/raw/` FAR-Trans CSVs (downloaded by `make data`, never committed; CC-BY 4.0)
- `dbt/` dbt-duckdb project: staging models, interactions with temporal split, feature tables
- `src/recs/` Python package
  - `data.py` loaders from the DuckDB warehouse
  - `candidates/` popularity, item-item, ALS, BPR candidate generators
  - `ranking/` point-in-time features and the LambdaRank ranker
  - `rules/` suitability rules (risk filter, sector cap)
  - `eval/` ranking metrics (Recall, NDCG, MAP), coverage, popularity bias, profitability@k
  - `pipelines/` runnable entry points used by the Makefile
- `api/` FastAPI service, `app/` Streamlit demo, `docs/` launch plan and notes

## Commands
- `make setup`, `make data`, `make dbt`, `make evaluate`, `make test`, `make lint`
- Use `uv run ...` for everything; never pip install into the venv.

## Rules that matter
- **No leakage.** Every feature is computed "as of" a cutoff date using only rows strictly
  before it. Splits are temporal (`config.VAL_START`, `config.TEST_START`), never random.
- A test interaction is a customer's first-time **Buy** of an asset in the window. Assets the
  customer already bought before the cutoff are excluded from targets and recommendations.
- Ranking metrics are computed only over customers with at least one target in the window.
- Keep functions pure where possible and covered by `tests/`. Tests use small synthetic frames
  and must not require the real dataset.
- Lint with ruff (line length 100). No em dashes in docs.
