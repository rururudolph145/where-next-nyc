# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hourly demand forecasting system for New York Taxi.

Data pipeline is structured using **medallion architecture** (Bronze -> Silver -> Gold)

The pipeline is designed as follows:
1. Data ingestion from API -> promote to Bronze layer
2. Perform data validation and cleaning -> promote to Silver layer
3. Perform feature engineering -> promote to Gold layer
4. Train and save demand prediction model
5. Use the saved demand prediction model to forecast taxi demand in specific location (h+3, h+6, h+12)

## Environment

This project uses **uv** for dependency and environment management (not pip/venv directly). Python 3.13+ required. PyTorch is pulled from the CUDA 12.6 index.

```bash
uv sync                  # install all dependencies into .venv
uv run python main.py    # run a script in the managed environment
uv add <package>         # add a new dependency to pyproject.toml
```

The `makefile` targets (`make setup`, `make run`) use a plain venv + pip workflow — these are kept for Docker/CI compatibility but local development should use `uv`.

## Common Commands

```bash
uv run python main.py          # run the full training pipeline (MLflow-tracked)
uv run pytest                  # run all tests
uv run pytest tests/test_clean.py::test_clean_data   # run a single test
uv run mlflow ui               # open MLflow experiment tracker
uv run uvicorn app:app --reload  # start the FastAPI serving endpoint (dev mode)
```

## Architecture

The project uses **pipeline-scoped modules** under `src/`, each with its own entrypoint and domain logic. Common code lives in `src/shared/`.

```
src/
  shared/          # centralized config loading
    config.py      # load_config() — single source of YAML config loading
  ingestion/       # Bronze → Silver pipeline
    ingest.py      # Ingestion class (Socrata API → Bronze parquet)
    clean.py       # DataCleaner class (Bronze → Silver)
  training/        # Model training pipeline
    trainer.py     # Trainer class (LGBMRegressor, saves to models/)
  inference/       # Prediction & evaluation pipeline
    predictor.py   # Predictor class (loads model, computes MAE/RMSE/R²)
  serving/         # FastAPI serving
    app.py         # FastAPI app with /predict endpoint
  utils/           # visualization helpers
    visualize.py
```

**Root entrypoints:**
- `main.py` — orchestrates the ingestion pipeline (ingest → clean, month by month)
- `app.py` — re-exports `src.serving.app.app` so `uvicorn app:app` works

**Ingestion pipeline** (orchestrated by `main.py`):
1. `src/ingestion/ingest.py` — downloads raw trip data from the Socrata API into the configured `data.bronze_path`
2. `src/ingestion/clean.py` — drops nulls, removes non-positive trip values, clips fare outliers via IQR, engineers time features (`hour`, `day_of_week`, `month`, `is_weekend`)

**Training:** `src/training/trainer.py` — trains the model from `config.yml` (currently `LGBMRegressor`), saves to `models/model.pkl`. The `MODELS` dict maps name strings to classes.

**Inference:** `src/inference/predictor.py` — loads `models/model.pkl`, evaluates with MAE / RMSE / R²

**Serving:** `src/serving/app.py` exposes a FastAPI endpoint at `POST /predict`. It loads `models/model.pkl` at startup, so the model must be trained before the server is started.

**Configuration:** All data paths, train/test split settings, and model hyperparameters live in `config.yml`. Configuration is loaded via `src.shared.config.load_config()`.

**Data:** Raw NYC TLC parquet files go under `data/`. The `data/` directory uses a medallion-style layout (e.g., `data/bronze/`). All `.parquet` and `.csv` files at any depth under `data/` are gitignored.

**Forecasting checklist:** `pipeline.MD` contains a detailed checklist of all pipeline stages (ingestion → monitoring). Use it to track implementation progress.

## Code Standards
- **Python 3.13** required
- **Mypy strict mode** enabled — all functions need type hints. use mypy strict mode
- **Ruff** for linting and formatting (line length: 88, docstring checks enabled)
- Documentation: docstrings in code should be in English
- Python functions must include Google-style docstrings
- Tests go in `tests/` directory using pytest

## Key Dependencies
- pandas, numpy: data manipulation
- scikit-learn: basic machine learning package
- mlflow: experiment tracking
- dvc: data versioning
- fastapi: api development
- evidently: model monitoring


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
5. **Code quality** — Ensure the code passes both linters before considering the work done:
   - `uv run mypy src/ --strict` — all type checks must pass
   - `uv run ruff check src/` — all lint rules must pass
   - `uv run ruff format --check src/` — formatting must be consistent

Do **not** skip or reorder these steps. If a step reveals issues, go back and fix before proceeding.

## Development Rules
- Secrets: Always put API keys and database credentials in .env
- App Settings: Put non-sensitive, environment-specific toggles in config.yaml or .config/.