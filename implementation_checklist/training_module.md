# Training Module — Implementation Checklist

## Requirement

Load Gold layer data, split by time, train LGBMRegressor, log to MLflow, save artifact.

**Input:** Gold parquet — hourly demand panel with features and `trip_count` target.
**Output:** Trained model saved to `models/model.pkl` + MLflow run with params/metrics/model.

## Implementation Steps

- [x] 1. Update `config.yml`: add `train.features` list, `train.split_ratios`, remove `shuffle`/`test_size`
- [x] 2. Rename `src/training/trainer.py` → `src/training/train.py`, rewrite `Trainer` class:
  - [x] 2a. `load_gold()` — load gold parquet from configured path
  - [x] 2b. `split_data()` — strict time-based 80/10/10 split on `hour` column
  - [x] 2c. `train()` — fit LGBMRegressor on training set
  - [x] 2d. `evaluate()` — compute MAE, RMSE, MAPE, WAPE on a given split
  - [x] 2e. `run()` — orchestrate: load → split → train → evaluate val & test → log MLflow → save model
- [x] 3. Add tests in `tests/test_train.py` (see test cases below)
- [x] 4. Pass `mypy --strict` and `ruff check` + `ruff format`

## Test Cases (`tests/test_train.py`)

### 1. `test_split_data_ratios`
- **What:** Verifies the 80/10/10 time-based split produces correct proportions.
- **Checks:**
  - Train set contains ~80% of the data.
  - Val set contains ~10% of the data.
  - Test set contains ~10% of the data.
  - All three sets are non-empty.
  - Total rows = train + val + test (no data lost).

### 2. `test_split_data_no_temporal_leakage`
- **What:** Verifies strict time ordering — no future data leaks into training.
- **Checks:**
  - Max `hour` in train < min `hour` in val.
  - Max `hour` in val < min `hour` in test.
  - No overlapping timestamps between any two splits.

### 3. `test_split_data_no_shuffle`
- **What:** Verifies the split is deterministic and order-preserving.
- **Checks:**
  - Each split is sorted by `hour`.
  - Calling `split_data()` twice produces identical results.

### 4. `test_train_model`
- **What:** Verifies the model is fitted after calling `train()`.
- **Checks:**
  - Model object has `fitted_` or `n_features_in_` attribute (LightGBM fitted indicator).
  - Model can produce predictions on a feature matrix.

### 5. `test_evaluate_metrics`
- **What:** Verifies `evaluate()` returns the correct metric keys and sensible values.
- **Checks:**
  - Returns dict with keys: `mae`, `rmse`, `mape`, `wape`.
  - All values are non-negative floats.
  - MAE <= RMSE (by definition).

### 6. `test_run_saves_model`
- **What:** Verifies `run()` saves a model artifact to the configured path.
- **Checks:**
  - `model.pkl` file exists at `model.store_path` after `run()`.
  - The saved file can be loaded back with joblib and produces predictions.

### 7. `test_run_logs_to_mlflow`
- **What:** Verifies `run()` creates an MLflow run with params and metrics.
- **Checks:**
  - An MLflow run is created (mock or local tracking).
  - Model hyperparameters are logged as params.
  - Val and test metrics (mae, rmse, mape, wape) are logged.
  - Model is logged as an MLflow artifact.
