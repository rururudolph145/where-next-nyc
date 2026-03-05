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

The project follows the MLOps template structure from [prsdm/mlops-project](https://github.com/prsdm/mlops-project), adapted for NYC yellow taxi demand forecasting.

**Pipeline flow** (`main.py` orchestrates):
1. `steps/ingest.py` — reads `data/train.parquet` and `data/test.parquet` using paths from `config.yml`
2. `steps/clean.py` — drops nulls, removes non-positive trip values, clips fare outliers via IQR, engineers time features (`hour`, `day_of_week`, `month`, `is_weekend`)
3. `steps/train.py` — trains the model from `config.yml` (currently `LGBMRegressor`), saves to `models/model.pkl`
4. `steps/predict.py` — loads `models/model.pkl`, evaluates with MAE / RMSE / R²

`main.py` — pipeline orchestration entry points
`tests/` — pytest tests

**Serving:** `app.py` exposes a FastAPI endpoint at `POST /predict`. It loads `models/model.pkl` at startup, so the model must be trained before the server is started.

**Configuration:** All data paths, train/test split settings, and model hyperparameters live in `config.yml`. Swap models by changing `model.name` — the `MODELS` dict in `steps/train.py` maps name strings to classes.

**Data:** Raw NYC TLC parquet files go under `data/`. The `data/` directory uses a medallion-style layout (e.g., `data/bronze/`). All `.parquet` and `.csv` files at any depth under `data/` are gitignored. Use `dataset.py` as the entry point for downloading/splitting source data.

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
- **Target column:** `trip_count` — features and target are separated in `Trainer.feature_target_separator()` and `Predictor.feature_target_separator()` by dropping this column.
- **MLflow model name:** registered as `ny_taxi_demand_model` in the local `mlruns/` store.
