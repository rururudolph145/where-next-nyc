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
├── steps/          # Pipeline steps (ingest, clean, train, predict)
├── data/           # Train / test datasets
├── models/         # Saved model artifacts
├── tests/          # Unit tests
├── docs/           # Project documentation
├── main.py         # Pipeline orchestration
├── app.py          # FastAPI serving endpoint
├── dataset.py      # Data extraction script
└── config.yml      # Hyperparameters and paths
```
