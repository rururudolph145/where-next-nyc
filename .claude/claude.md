# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NYC yellow-taxi demand forecasting system** — a personal portfolio project demonstrating an end-to-end MLOps pipeline on a real-world public dataset.

### The Problem
NYC yellow-taxi drivers spend roughly a third of their shift empty — circling, idling, or returning from outer-borough drop-offs to areas they *guess* will be busy. Experienced drivers already know Midtown is busy at 5 pm; the hard question is the one they can't answer from intuition: **where will demand be 30 minutes from now**, when they finish the current drop-off.

### What This System Does
Predicts pickup volume per taxi zone for the next 30 minutes, refreshed every 15 minutes, and serves the result as a ranked heatmap a driver app can render in one API call.

- **Audience:** Drivers / fleet dispatchers deciding *where to position next*.
- **Decision horizon:** 30 minutes — roughly the time to finish a trip and reposition.
- **Resolution:** Per taxi zone (263 zones), per 15-minute bucket.
- **Counterfactual claim:** Beat what an experienced driver already knows (the obvious time-of-day pattern), measured against a *last-week-same-hour* baseline — not against zero.

### How Value Is Measured
Model-quality metrics (MAE / RMSE / WAPE) measure the model. Two bridging metrics tie ML output back to the business question:

- **Recall @ top-K zones** — of the trips that actually happened, what fraction occurred in the top-K predicted zones?
- **WAPE lift over baseline** — how much better is LightGBM than "last week, same hour, same zone"?

### Pipeline Shape
Medallion architecture (Bronze → Silver → Gold), with monitoring closing the loop:

1. Ingest raw NYC TLC trips → Bronze
2. Validate schema + quality, clean → Silver
3. Feature engineering (lags, cyclical time, k-NN spatial smoothing) → Gold
4. Train LightGBM model with time-ordered split, log to MLflow, gate on validation
5. Serve `/heatmap` via FastAPI, refreshed every 15 minutes by Airflow
6. Monitor drift + actuals, trigger retrain on degradation

### Honest Limitations (worth reflecting in any agent-generated docs)
- TLC publishes trip records on a 1–2 month lag → the 15-min refresh cadence is *simulated*, not live.
- Single-agent assumption: if every driver follows the same heatmap, supply outruns demand in the predicted zones (self-defeating prediction).
- Macro patterns only — no weather, events, or transit disruptions yet.

### Portfolio Goals
- Showcase a clean **medallion architecture** data pipeline with explicit data-validation and feature-store seams
- Demonstrate **MLflow** experiment tracking and a model registry with staging/production stages
- Provide a working **FastAPI** `/heatmap` endpoint with a refreshable, cache-backed sub-200ms response
- Show a closed-loop **monitoring** lane (prediction log → actuals join → drift → retrain trigger)
- Maintain **production-quality** standards (strict typing, tests, lint/format) despite being a personal project

## Environment

This project uses **uv** for dependency and environment management (not pip/venv directly). **Python 3.13+ required**. PyTorch is pulled from the CUDA 12.6 index.

```bash
uv sync                          # install all dependencies into .venv
uv run python -m pipeline.run_all  # run a script in the managed environment
uv add <package>                 # add a new dependency to pyproject.toml
```

The `makefile` targets (`make setup`, `make run`) use a plain venv + pip workflow — these are kept for Docker/CI compatibility but local development should use `uv`.

## Tech Stack

### Core
- **Python 3.13+** (required)
- Package management: **uv** (`pyproject.toml` + `uv.lock`)
- Linter / formatter: **ruff** (line-length=88, docstring checks enabled)
- Type checking: **mypy** (`uv run mypy app tests`; `orchestration/` is excluded — see `pyproject.toml`)
- Tests: **pytest**

### Key Dependencies
- **pandas, numpy** — data manipulation
- **scikit-learn** — basic machine learning utilities
- **lightgbm** — `LGBMRegressor` is the current default model
- **mlflow** — experiment tracking (local `mlruns/` store) + model registry
- **pandera** — schema/quality validation at the ingest gate
- **geopandas** — taxi-zone shapefile → zone centroids for the heatmap
- **fastapi** + **uvicorn** — serving API
- Monitoring is **hand-rolled** in `app/monitoring/` (prediction log + actuals join), not a third-party tool

### Interface
There is **no frontend**. The FastAPI service is the only deliverable interface — consume `GET /heatmap` via `curl`, the Swagger UI at `/docs`, or any HTTP client. The service also exposes `GET /health` (data freshness / simulated-now) and `GET /model-info` (Production model provenance).

## Common Commands

```bash
uv run python -m pipeline.run_all                      # run the full ingestion pipeline (MLflow-tracked)
uv run pytest                                          # run all tests
uv run pytest tests/test_serving.py                    # run a single test file
uv run pytest -k "watermark"                           # run tests matching an expression
uv run mlflow ui                                       # open MLflow experiment tracker
make demo                                              # serve FastAPI on :8000 (uvicorn app.api.main:app)
uv run ruff check . && uv run ruff format --check .    # lint + format check (CI gate)
uv run mypy app tests                                  # type check (orchestration/ excluded)
```

## Architecture

The project uses **pipeline-scoped modules** under `app/`, each with its own domain logic. Common code lives in `app/core/`. Pipeline orchestration scripts live in the repo-root `pipeline/` directory.

```
NewYorkTaxiDemandForecasting/
├── .claude/                  # Claude Code settings
├── app/
│   ├── core/                 # centralized config loading + logging setup
│   │   ├── config.py         #   load_config() — single source of YAML config loading
│   │   └── logging.py        #   setup_logging() — shared logger configuration
│   ├── ingestion/            # Bronze → Silver → Gold pipeline
│   │   ├── ingest.py         #   Ingestion class (local TLC parquet → Bronze parquet)
│   │   ├── clean.py          #   DataCleaner class (Bronze → Silver)
│   │   └── feature.py        #   FeatureEngineer class (Silver → Gold)
│   ├── training/             # Model training pipeline
│   │   └── trainer.py        #   Trainer class (LGBMRegressor, saves to models/)
│   ├── inference/            # Prediction & evaluation pipeline
│   │   ├── predictor.py      #   Predictor class (loads model, computes MAE/RMSE/R²)
│   │   └── simulator.py      #   hourly inference simulator
│   └── api/                  # FastAPI serving
│       └── main.py           #   FastAPI app with GET /heatmap endpoint
├── pipeline/                 # numbered runner scripts + run_all entrypoint
├── tests/                    # pytest tests
├── data/                     # medallion layout (e.g., data/bronze/); .parquet/.csv are gitignored
├── models/                   # trained models (model.pkl)
├── mlruns/                   # MLflow local store
├── implementation_checklist/ # per-feature checklist files (see Development Workflow)
├── config.yml                # data paths, train/test split, model hyperparameters
├── pipeline.MD               # detailed checklist of all pipeline stages
├── pyproject.toml
├── uv.lock
├── dockerfile                # builds the FastAPI serving image
└── makefile                  # Docker/CI-compatible setup/run targets
```

**Entrypoints:**
- `pipeline.run_all` — orchestrates the full pipeline (ingest → clean → feature → train → infer)
- `app.api.main:app` — FastAPI ASGI app served via `uvicorn app.api.main:app`

**Ingestion pipeline** (orchestrated by `pipeline.run_all`):
1. `app/ingestion/ingest.py` — reads raw NYC TLC parquet (fixed 2021 Q1 window in `config.yml`) into the configured `data.bronze_path`. Under the Airflow demo it is a **cumulative** ingest: each run extends the window `[seed_start, window_end)` and bronze grows monotonically. There is no live/Socrata fetch.
2. `app/ingestion/clean.py` — drops nulls, removes non-positive trip values, clips fare outliers via IQR, engineers time features (`hour`, `day_of_week`, `month`, `is_weekend`)
3. `app/ingestion/feature.py` — produces Gold-layer features (lags, rolling windows, cyclical encoding)

**Training:** `app/training/trainer.py` — trains the model from `config.yml` (currently `LGBMRegressor`), saves to `models/model.pkl`. The `MODELS` dict maps name strings to classes.

**Inference:** `app/inference/predictor.py` — loads `models/model.pkl`, evaluates with MAE / RMSE / R².

**Serving:** `app/api/main.py` exposes `GET /heatmap` (plus `GET /health` for data freshness / simulated-now and `GET /model-info` for Production model provenance). It loads pre-computed predictions and the zone lookup into memory at startup, so the inference simulation must be run before the server is started.

**Configuration:** All data paths, train/test split settings, and model hyperparameters live in `config.yml`. Configuration is loaded via `app.core.config.load_config()`.

**Data:** Raw NYC TLC parquet files go under `data/`. The `data/` directory uses a medallion-style layout (e.g., `data/bronze/`). All `.parquet` and `.csv` files at any depth under `data/` are gitignored.

**Forecasting checklist:** `pipeline.MD` contains a detailed checklist of all pipeline stages (ingestion → monitoring). Use it to track implementation progress.

**Clock abstraction (hard invariant):** `app/core/clock.py` is the **only** legitimate caller of `datetime.now(tz=...)`. Everything else takes a `Clock` (Protocol) — `WallClock` in production, `SimulatedClock` in tests. Do not reintroduce direct `datetime.utcnow()` / `time.sleep` calls in application code.

**The watermark IS the simulation position.** `data/watermarks/refresh_15min/.current_watermark` is a control file, not derived from a clock. Each demo DAG run reads it, ingests the next 15-min slice, advances it, and writes it back. `/health` reads it for `simulated_now`. In `config.yml`, `ingest.slice_duration` MUST equal `feature.bucket` (both `15min`). Timestamps are **tz-naive NYC wall-clock** (`tpep_pickup_datetime` has no `Z`) — do not attach UTC.

**Orchestration layer:** the Airflow stack lives under `orchestration/` (Airflow 3.2.2 via docker-compose). **`app/` MUST NOT import from `orchestration/`** — enforced by `tests/test_pbi25a_orchestrator_skeleton.py::test_app_does_not_import_from_orchestration`. DAG business logic lives in Airflow-free, host-testable `orchestration/lib/*_helpers.py`; DAG files are thin wrappers. `orchestration_experiment/` is a separate incremental-build sandbox (Airflow 2.10.3), not production. Run it with `make airflow-up` then repeat `make demo-tick` to walk the timeline one 15-min slice per trigger. If a DAG vanishes from the UI, first run `docker compose -f orchestration/docker-compose.yaml exec airflow-scheduler airflow dags list-import-errors` — Airflow swallows import errors.

**MLflow DB caveat:** never reuse an `mlflow.db` created on the host inside the container — its `experiments.artifact_location` bakes a host absolute path that is unreachable in-container. Delete `data/mlflow.db` before the first container training run and let MLflow recreate it (`model.pkl` on disk is unaffected).

## Code Standards
- **Python 3.13** required
- **Mypy** — type-check with `uv run mypy app tests` (config in `pyproject.toml`; `orchestration/` is excluded)
- **Ruff** for linting and formatting (line length: 88, docstring checks enabled)
- Docstrings in code should be in **English**
- Python functions must include **Google-style** docstrings
- Tests go in the `tests/` directory using pytest

## Key Decisions

- **Time-series split:** Use time-based splits only — never shuffle for temporal data (see `pipeline.MD` §6).
- **Target column:** `target_sum_t1_t2` (set in `config.yml` under `train.target`) — a 30-min lookahead = sum of the next two 15-min bins. Not raw `trip_count`.
- **MLflow model name:** registered as `ny_taxi_demand_model` in the local `mlruns/` store.

## Development Workflow

Every new feature or pipeline step **must** follow this workflow in order:

1. **Requirement definition** — Clearly define what the feature does, its inputs/outputs, and acceptance criteria before writing any code.
2. **Implementation checklist** — Create a markdown checklist file in `implementation_checklist/` (e.g. `implementation_checklist/feature_engineering.md`) that breaks the work into concrete, verifiable steps.
3. **Implementation** — Implement the feature following the checklist step by step. Check off items as they are completed.
4. **Testing** — Add or update test cases in `tests/` to cover the new functionality. All tests must pass (`uv run python -m pytest`).
5. **Code quality** — Ensure the code passes all linters before considering the work done:
   - `uv run mypy app tests` — all type checks must pass
   - `uv run ruff check .` — all lint rules must pass
   - `uv run ruff format --check .` — formatting must be consistent

Do **not** skip or reorder these steps. If a step reveals issues, go back and fix before proceeding.

## Development Rules

- **Secrets:** Always put API keys and database credentials in `.env`. Never commit `.env`.
- **App settings:** Put non-sensitive, environment-specific toggles in `config.yml`.
- **Data hygiene:** Never commit `.parquet` / `.csv` files under `data/` (all are gitignored at any depth).
- **No force-kill:** Do not run `pkill` or `kill -9` (can disconnect VSCode etc.). Ask the user to restart servers.
- **Branch/PR flow:** `main` ← `dev_MVP` (release PRs only) ← `feature/*` / `bug_fix_*`. **Never** PR a feature branch directly to `main` — always `gh pr create --base dev_MVP`. One commit = one logical change; emoji prefixes (✨ feat, 🐛 fix, ♻️ refactor, 📝 docs, 💚 CI, 🙈 chore).

The authoritative, detailed rules live in `.claude/rules/` (coding-style, testing, git-workflow, security, performance, agents) — consult them for anything not covered here.

## Commit Operation Rule

**Before every commit, review `CLAUDE.md` and ensure consistency before committing.**

1. Just before creating a commit, read `CLAUDE.md` and verify that the facts changed by the commit (project goals, architecture, directory structure, tech stack, key decisions, etc.) are consistent with it.
2. If there are inconsistencies, outdated descriptions, or missing items, update `CLAUDE.md`.
3. If updates occur, **include the change to `CLAUDE.md` in the same commit** (do not split into a separate commit).
4. If no update is needed, briefly note "CLAUDE.md review: no update needed" in the commit/report.

### Why
This repository is intended as a portfolio piece, so its specification is recorded not only in code but also in `CLAUDE.md`. Because every Claude Code session loads `CLAUDE.md` first, an outdated file will skew every subsequent judgement. Keeping it in sync prevents drift over time.

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/plan` | Plan an implementation |
| `/code-review` | Code review |
| `/build-fix` | Resolve build errors |
| `/log-analyze` | Log analysis |
| `/refactor` | Refactoring |
| `/test` | Test generation / execution |
| `/tdd` | Test-driven development |
| `/deploy-check` | Pre-deployment check |
