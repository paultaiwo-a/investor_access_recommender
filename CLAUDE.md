# CLAUDE.md

Notes for working in this repo.

## What this is

A small recommender on the FAR-Trans dataset. SQL loads the csvs into DuckDB and labels buys by date. Python fits popularity and item-item, then scores them with recall and nDCG.

## Layout

- `data/raw/` FAR-Trans csvs. Downloaded by `make data`, not committed.
- `sql/prepare.sql` cleaning and the date split.
- `src/recs/` load the warehouse, build a snapshot, fit the two models, score them.
- `tests/` made-up data. No download needed.

## Commands

- `make setup`, `make data`, `make evaluate`, `make test`, `make lint`
- Use `uv run ...`. Do not pip install into the venv.

## Rules that matter

- No leakage. History is buys strictly before the cutoff. The test window is 30 May 2022 to 30 Nov 2022.
- A target is a customer's first buy of that asset in the window. Assets they already bought are left off the list and do not count as hits.
- Scores are only for customers who bought something new in the window.
- Tests use made-up frames. They must not need the real dataset.
- Lint with ruff, line length 100. No em dashes in docs.
