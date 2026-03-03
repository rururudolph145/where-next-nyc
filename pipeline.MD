# Minimum Demand Forecasting Pipeline Checklist

## 1. Data Ingestion
- [ ] Identify and document all data sources (transactional, IoT, ERP, external APIs)
- [ ] Define ingestion frequency (batch vs. streaming) and schedule
- [ ] Implement connectors/adapters for each source
- [ ] Validate raw data arrives within acceptable latency SLA
- [ ] Set up dead-letter queue or error log for failed ingestion events

## 2. Data Storage
- [ ] Choose raw storage layer (data lake, object store, RDBMS)
- [ ] Define partitioning strategy (e.g., by date, region, SKU)
- [ ] Enable append-only writes to preserve raw history
- [ ] Enforce retention policy and archival rules
- [ ] Confirm sufficient storage capacity for historical lookback window

## 3. Data Validation & Quality
- [ ] Schema validation (expected columns, data types)
- [ ] Null / missing value checks on target variable and key features
- [ ] Range / outlier detection (e.g., negative demand values)
- [ ] Duplicate record detection and deduplication logic
- [ ] Referential integrity checks (e.g., unknown SKU/location codes)
- [ ] Data freshness check — alert if source data is stale

## 4. Data Preprocessing
- [ ] Normalize datetime to a single timezone / calendar grain (hourly, daily, weekly)
- [ ] Aggregate raw transactions to forecast granularity
- [ ] Handle missing time steps (zero-fill vs. forward-fill vs. interpolation — document choice)
- [ ] Cap or clip extreme outliers with documented business rules
- [ ] Encode categorical features (SKU, location, category)
- [ ] Merge external signals (weather, holidays, promotions, macroeconomic indicators)

## 5. Feature Engineering
- [ ] Create lag features (t-1, t-7, t-28, etc.) appropriate to forecast horizon
- [ ] Create rolling statistics (mean, std, min, max) over relevant windows
- [ ] Add calendar features (day-of-week, month, week-of-year, is_holiday)
- [ ] Add trend and seasonality decomposition features if needed
- [ ] Document feature lineage and transformation logic
- [ ] Verify no data leakage (future data visible to training window)

## 6. Train / Validation Split
- [ ] Use time-based split (no random shuffle for time-series data)
- [ ] Define warm-up period to avoid cold-start bias
- [ ] Define validation window size relative to forecast horizon
- [ ] Optionally implement walk-forward / expanding-window cross-validation
- [ ] Document cutoff dates for reproducibility

## 7. Model Training
- [ ] Select baseline model (e.g., naive seasonal, moving average)
- [ ] Select candidate model(s) (e.g., LightGBM, XGBoost, Prophet, ARIMA, LSTM)
- [ ] Define hyperparameter search space and tuning strategy
- [ ] Log all training runs with parameters, metrics, and artifact locations
- [ ] Store trained model artifacts in a versioned model registry
- [ ] Record training data version alongside model version

## 8. Model Evaluation
- [ ] Define primary metric (e.g., MAE, RMSE, MAPE, sMAPE, wMQL)
- [ ] Evaluate on out-of-sample holdout set
- [ ] Compare against baseline — model must beat baseline to proceed
- [ ] Segment evaluation by key dimensions (region, category, SKU tier)
- [ ] Check for systematic bias (persistent over- or under-forecast)
- [ ] Document acceptable error thresholds per segment

## 9. Inference / Scoring
- [ ] Ensure feature pipeline used at inference is identical to training pipeline
- [ ] Version-lock feature transformations alongside model artifact
- [ ] Implement batch or real-time scoring endpoint as required
- [ ] Output forecast with prediction intervals (min/max or quantiles) when possible
- [ ] Write forecasts to a downstream store with metadata (model version, run timestamp)
- [ ] Handle cold-start cases (new SKUs / locations with no history)

## 10. Monitoring & Alerting
- [ ] Track input data drift (distribution shift in features)
- [ ] Track prediction drift (shift in forecast distribution)
- [ ] Monitor forecast accuracy on actuals as they arrive (rolling MAE/MAPE)
- [ ] Set alert thresholds for accuracy degradation
- [ ] Alert on pipeline failures, late data, or schema changes
- [ ] Publish monitoring dashboard for stakeholders

## 11. Retraining & Lifecycle
- [ ] Define retraining trigger (scheduled, drift-based, or accuracy-based)
- [ ] Automate retraining pipeline with the same validation gates as initial training
- [ ] Implement champion/challenger model promotion logic
- [ ] Archive retired model versions (do not delete — needed for audit)
- [ ] Document model deprecation and rollback procedure

## 12. Infrastructure & Orchestration
- [ ] Choose orchestration tool (Airflow, Prefect, Dagster, Step Functions, etc.)
- [ ] Define DAG / workflow with clear task dependencies
- [ ] Configure retries and failure notifications for each task
- [ ] Parameterize pipeline for different horizons, granularities, and geographies
- [ ] Ensure idempotent runs (re-running a pipeline step produces the same result)
- [ ] Document environment and dependency versions (requirements.txt / pyproject.toml)

## 13. Security & Compliance
- [ ] Restrict access to raw data to authorized roles only
- [ ] Mask or exclude PII fields not needed for forecasting
- [ ] Audit log all data access and model predictions
- [ ] Comply with data residency requirements for each region
- [ ] Review third-party data license terms before use

## 14. Documentation & Handoff
- [ ] Document data dictionary for all input features and output columns
- [ ] Record business assumptions embedded in preprocessing rules
- [ ] Provide runbook for on-call engineers (how to debug, restart, rollback)
- [ ] Store pipeline code in version control with meaningful commit history
- [ ] Schedule periodic review cadence with business stakeholders
