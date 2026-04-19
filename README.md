# New York Taxi Demand Forecasting

An end-to-end MLOps project that forecasts NYC yellow taxi trip demand using LightGBM, tracked with MLflow, served via FastAPI, and monitored with Evidently AI.

Current data cleaning strategy:
- Remove anomalous pickups (pickup location outside of New York)
- Remove negative-duration trips (dropoff before pickup), as these are corrupt records

Other anomalous data (speed, distance, fare outliers) is out of scope as they are deemed not necessarily related to predicting demand

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
