# PBI 2 — Inference Pipeline: Implementation Checklist

## Requirement

Build pipeline step that loads a trained model and generates batch predictions from Gold data.

**Input:** Trained model (`models/model.pkl`) + Gold layer features.
**Output:** Predictions table (`data/predictions/demand_predictions.parquet`) with columns: `pulocationid`, `hour`, `predicted_trip_count`, `actual_trip_count`.

## Implementation Steps

- [x] 1. Add `inference.output_path` to `config.yml`
- [x] 2. Rewrite `src/inference/predictor.py` — `Predictor` class:
  - [x] 2a. `__init__()` — load config, model, features, target
  - [x] 2b. `load_model()` — from `models/model.pkl`
  - [x] 2c. `load_gold()` — from configured `gold_path`
  - [x] 2d. `predict()` — run model on `df[features]`, return predictions array
  - [x] 2e. `build_predictions_table()` — `(pulocationid, hour, predicted_trip_count, actual_trip_count)`
  - [x] 2f. `evaluate()` — MAE, RMSE, MAPE, WAPE (same as Trainer)
  - [x] 2g. `save_predictions()` — write parquet to `inference.output_path`
  - [x] 2h. `run()` — orchestrate all steps, return metrics + output_path
- [x] 3. Create `pipeline/05_inference.py` — follows `04_train.py` pattern
- [x] 4. Add tests in `tests/test_inference.py`
- [x] 5. Pass `mypy --strict` and `ruff check` + `ruff format`

## Test Cases (`tests/test_inference.py`)

### 1. `TestLoadModel`
- Loads a saved model successfully.
- Raises error on missing model file.

### 2. `TestPredict`
- Returns predictions with correct output shape.
- All predictions are finite values.

### 3. `TestBuildPredictionsTable`
- Output has correct columns: `pulocationid`, `hour`, `predicted_trip_count`, `actual_trip_count`.
- Row count matches input data.
- Zone IDs are preserved.

### 4. `TestEvaluate`
- Returns dict with keys: `mae`, `rmse`, `mape`, `wape`.
- All values are non-negative.

### 5. `TestSavePredictions`
- Parquet file is created at configured path.
- File can be read back and matches original data.

### 6. `TestRun`
- End-to-end: returns dict with `metrics`, `output_path`, `n_predictions`.
- Predictions file exists after run.
