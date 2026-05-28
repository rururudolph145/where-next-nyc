# New York Taxi Demand Forecasting

An end-to-end MLOps project for forecasting NYC yellow taxi trip demand by location and time.

## Overview

This project covers the full ML lifecycle:

- **Data ingestion** — raw NYC TLC parquet files
- **Data cleaning** — feature engineering from trip timestamps
- **Model training** — LightGBM regressor tracked with MLflow
- **Model serving** — FastAPI REST API deployed via Docker
- **Monitoring** — data drift and prediction drift via Evidently AI

## Quickstart

```bash
make setup
make run
```

## Project Structure

```
├── app/            # Library code (ingestion, training, inference, api, core)
├── pipeline/       # Numbered pipeline runner scripts + run_all entrypoint
├── data/           # Bronze / Silver / Gold datasets
├── models/         # Saved model artifacts
├── tests/          # Unit tests
├── docs/           # Project documentation
└── config.yml      # Hyperparameters and paths
```
