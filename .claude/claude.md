# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hourly demand forecasting system for New York Taxi** — a personal portfolio project to demonstrate an end-to-end MLOps pipeline (data ingestion → model serving → monitoring) on a real-world public dataset.

The data pipeline is structured using **medallion architecture** (Bronze → Silver → Gold):

1. Data ingestion from API → promote to Bronze layer
2. Data validation and cleaning → promote to Silver layer
3. Feature engineering → promote to Gold layer
4. Train and save the demand prediction model
5. Use the saved model to forecast taxi demand at specific locations (h+3, h+6, h+12)

### Portfolio Goals
- Showcase a clean **medallion architecture** data pipeline
- Demonstrate **MLflow** experiment tracking and **DVC** data versioning
- Provide a working **FastAPI** prediction endpoint as the deployment artifact
- Maintain **production-quality** standards (strict typing, tests, lint/format) despite being a personal project

## Environment

This project uses **uv** for dependency and environment management (not pip/venv directly). **Python 3.13+ required**. PyTorch is pulled from the CUDA 12.6 index.

```bash
uv sync                  # install all dependencies into .venv
uv run python main.py    # run a script in the managed environment
uv add <package>         # add a new dependency to pyproject.toml
```

The `makefile` targets (`make setup`, `make run`) use a plain venv + pip workflow — these are kept for Docker/CI compatibility but local development should use `uv`.

## Tech Stack

### Core
- **Python 3.13+** (required)
- Package management: **uv** (`pyproject.toml` + `uv.lock`)
- Linter / formatter: **ruff** (line-length=88, docstring checks enabled)
- Type checking: **mypy strict mode** — all functions need type hints
- Tests: **pytest**

### Key Dependencies
- **pandas, numpy** — data manipulation
- **scikit-learn** — basic machine learning utilities
- **lightgbm** — `LGBMRegressor` is the current default model
- **mlflow** — experiment tracking (local `mlruns/` store)
- **dvc** — data versioning
- **fastapi** + **uvicorn** — serving API
- **evidently** — model monitoring

### Frontend (planned, not yet implemented)
A frontend layer is **planned for a later phase** to visualize predictions (e.g., heatmap of taxi demand by zone × hour, comparison of forecast vs actuals). When added, the intended stack is:

- **Vite + React + TypeScript** — SPA build tooling
- **@tanstack/react-query** — server-state management for FastAPI calls
- **react-router-dom** — routing
- Minimal styling via **CSS Modules** + a single `styles.css` (no heavy UI framework)
- During development, `vite.config.ts` will proxy `/api` → `http://127.0.0.1:8000` to avoid CORS
- Planned location: `frontend/` directory at the repo root, with `src/{api,components,pages,types,...}`

Until the frontend is built, the FastAPI `POST /predict` endpoint is the only deliverable interface (consume via `curl`, Swagger UI at `/docs`, or any HTTP client).

## Common Commands

```bash
uv run python main.py                                # run the full ingestion pipeline (MLflow-tracked)
uv run pytest                                        # run all tests
uv run pytest tests/test_clean.py::test_clean_data   # run a single test
uv run mlflow ui                                     # open MLflow experiment tracker
uv run uvicorn app:app --reload                      # start the FastAPI serving endpoint (dev mode)
uv run ruff check src/ && uv run ruff format src/    # lint + format
uv run mypy src/ --strict                            # type check (strict)
```

## Architecture

The project uses **pipeline-scoped modules** under `src/`, each with its own entrypoint and domain logic. Common code lives in `src/shared/`.

```
NewYorkTaxiDemandForecasting/
├── .claude/                  # Claude Code settings
├── src/
│   ├── shared/               # centralized config loading
│   │   └── config.py         #   load_config() — single source of YAML config loading
│   ├── ingestion/            # Bronze → Silver pipeline
│   │   ├── ingest.py         #   Ingestion class (Socrata API → Bronze parquet)
│   │   └── clean.py          #   DataCleaner class (Bronze → Silver)
│   ├── training/             # Model training pipeline
│   │   └── trainer.py        #   Trainer class (LGBMRegressor, saves to models/)
│   ├── inference/            # Prediction & evaluation pipeline
│   │   └── predictor.py      #   Predictor class (loads model, computes MAE/RMSE/R²)
│   ├── serving/              # FastAPI serving
│   │   └── app.py            #   FastAPI app with /predict endpoint
│   └── utils/                # visualization helpers
│       └── visualize.py
├── tests/                    # pytest tests
├── data/                     # medallion layout (e.g., data/bronze/); .parquet/.csv are gitignored
├── models/                   # trained models (model.pkl)
├── mlruns/                   # MLflow local store
├── implementation_checklist/ # per-feature checklist files (see Development Workflow)
├── config.yml                # data paths, train/test split, model hyperparameters
├── pipeline.MD               # detailed checklist of all pipeline stages
├── main.py                   # orchestrates the ingestion pipeline (ingest → clean, month by month)
├── app.py                    # re-exports src.serving.app.app so `uvicorn app:app` works
├── pyproject.toml
├── uv.lock
└── makefile                  # Docker/CI-compatible setup/run targets
```

**Root entrypoints:**
- `main.py` — orchestrates the ingestion pipeline (ingest → clean, month by month)
- `app.py` — re-exports `src.serving.app.app` so `uvicorn app:app` works

**Ingestion pipeline** (orchestrated by `main.py`):
1. `src/ingestion/ingest.py` — downloads raw trip data from the Socrata API into the configured `data.bronze_path`
2. `src/ingestion/clean.py` — drops nulls, removes non-positive trip values, clips fare outliers via IQR, engineers time features (`hour`, `day_of_week`, `month`, `is_weekend`)

**Training:** `src/training/trainer.py` — trains the model from `config.yml` (currently `LGBMRegressor`), saves to `models/model.pkl`. The `MODELS` dict maps name strings to classes.

**Inference:** `src/inference/predictor.py` — loads `models/model.pkl`, evaluates with MAE / RMSE / R².

**Serving:** `src/serving/app.py` exposes a FastAPI endpoint at `POST /predict`. It loads `models/model.pkl` at startup, so the model must be trained before the server is started.

**Configuration:** All data paths, train/test split settings, and model hyperparameters live in `config.yml`. Configuration is loaded via `src.shared.config.load_config()`.

**Data:** Raw NYC TLC parquet files go under `data/`. The `data/` directory uses a medallion-style layout (e.g., `data/bronze/`). All `.parquet` and `.csv` files at any depth under `data/` are gitignored.

**Forecasting checklist:** `pipeline.MD` contains a detailed checklist of all pipeline stages (ingestion → monitoring). Use it to track implementation progress.

## Code Standards
- **Python 3.13** required
- **Mypy strict mode** enabled — all functions need type hints. Use `uv run mypy src/ --strict`
- **Ruff** for linting and formatting (line length: 88, docstring checks enabled)
- Docstrings in code should be in **English**
- Python functions must include **Google-style** docstrings
- Tests go in the `tests/` directory using pytest

## Key Decisions

- **Time-series split:** Use time-based splits only — never shuffle for temporal data (see `pipeline.MD` §6).
- **Target column:** `trip_count` — features and target are separated in `src.training.trainer.Trainer.feature_target_separator()` and `src.inference.predictor.Predictor.feature_target_separator()` by dropping this column.
- **MLflow model name:** registered as `ny_taxi_demand_model` in the local `mlruns/` store.

## Development Workflow

Every new feature or pipeline step **must** follow this workflow in order:

1. **Requirement definition** — Clearly define what the feature does, its inputs/outputs, and acceptance criteria before writing any code.
2. **Implementation checklist** — Create a markdown checklist file in `implementation_checklist/` (e.g. `implementation_checklist/feature_engineering.md`) that breaks the work into concrete, verifiable steps.
3. **Implementation** — Implement the feature following the checklist step by step. Check off items as they are completed.
4. **Testing** — Add or update test cases in `tests/` to cover the new functionality. All tests must pass (`uv run python -m pytest`).
5. **Code quality** — Ensure the code passes all linters before considering the work done:
   - `uv run mypy src/ --strict` — all type checks must pass
   - `uv run ruff check src/` — all lint rules must pass
   - `uv run ruff format --check src/` — formatting must be consistent

Do **not** skip or reorder these steps. If a step reveals issues, go back and fix before proceeding.

## Development Rules

- **Secrets:** Always put API keys (e.g., Socrata app token) and database credentials in `.env`. Never commit `.env`.
- **App settings:** Put non-sensitive, environment-specific toggles in `config.yml` or `.config/`.
- **Data hygiene:** Never commit `.parquet` / `.csv` files under `data/`. Use DVC for data versioning.
- **No force-kill:** Do not run `pkill` or `kill -9` (can disconnect VSCode etc.). Ask the user to restart servers.

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
