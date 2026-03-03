# New York Taxi Demand Forecasting

An end-to-end MLOps project that forecasts NYC yellow taxi trip demand using LightGBM, tracked with MLflow, served via FastAPI, and monitored with Evidently AI.

## Setup

```bash
make setup
```

## Run

```bash
make run
```

## MLflow UI

```bash
make mlflow
```

## Test

```bash
make test
```

## Docker

```bash
docker build -t ny-taxi-demand .
docker run -p 80:80 ny-taxi-demand
```
