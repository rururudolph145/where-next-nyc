# PBI 27 — `/health` + `/model-info` + Simulated-Clock Flag

## Why This Matters

The README claims the API serves replayed historical data with a versioned, baseline-beating model. Right now there is no machine-readable way to verify any of that. Operators and downstream consumers have no signal for whether predictions are stale, which model version is live, or whether the "simulated" disclaimer applies to the current response. This PBI closes that gap.

---

## User Story

As an API consumer, I want dedicated endpoints that report the health of the prediction service and the current model's provenance, so that I can verify data freshness, confirm the simulated nature of the feed, and inspect model quality without reading logs or source code.

---

## Acceptance Criteria

### `GET /health` — happy path

1. Given the predictions parquet is loaded and a Production model version exists in the registry, when `GET /health` is called, the response is HTTP 200 with the following JSON shape and types:

   ```json
   {
     "status": "ok",
     "simulated": true,
     "data_publication_lag_days": 42,
     "last_prediction_bucket": "2024-03-15T23:45:00",
     "model_version": "3"
   }
   ```

   - `status`: string, always `"ok"` on the happy path.
   - `simulated`: boolean, always `true` (hardcoded for MVP).
   - `data_publication_lag_days`: non-negative integer, computed as `(today_utc - max(bucket_time)).days` from the loaded predictions parquet.
   - `last_prediction_bucket`: ISO-8601 string, equal to `max(hour)` across the loaded predictions parquet.
   - `model_version`: string, the version number of the current Production model from the registry.

2. `data_publication_lag_days` is never negative. When `max(bucket_time)` is in the future relative to `today_utc`, the value returned is `0`.

3. `data_publication_lag_days` is `null` when the predictions parquet is loaded but contains zero rows (empty file).

### `GET /health` — degraded (no Production model)

4. Given the predictions parquet is loaded but no Production model version exists in the registry, when `GET /health` is called, the response is HTTP 200 with:

   ```json
   {
     "status": "degraded",
     "simulated": true,
     "data_publication_lag_days": 42,
     "last_prediction_bucket": "2024-03-15T23:45:00",
     "model_version": null
   }
   ```

   - `status` is `"degraded"`.
   - `model_version` is `null`.
   - All prediction-parquet-derived fields are still populated normally.

### `GET /health` — predictions unavailable

5. Given the predictions parquet was not loaded at startup (file absent), when `GET /health` is called, the response is HTTP 503 with a `detail` field explaining that predictions are not available. This matches the existing 503 behaviour on `/heatmap`.

### `GET /model-info` — happy path

6. Given a Production model version exists in the registry, when `GET /model-info` is called, the response is HTTP 200 with the following JSON shape:

   ```json
   {
     "status": "ok",
     "model_version": "3",
     "stage": "Production",
     "training_date": "2024-03-10T08:22:01",
     "model_test_wape": 0.1823,
     "wape_lift_pct": 12.4
   }
   ```

   - `model_version`: string, the Production version number.
   - `stage`: string, always `"Production"` when a Production version is present.
   - `training_date`: ISO-8601 string derived from the version's `creation_timestamp` (milliseconds since epoch, converted to UTC).
   - `model_test_wape`: float or `null`, read from the `model_test_wape` version tag; `null` if the tag is absent.
   - `wape_lift_pct`: float or `null`, read from the `wape_lift_pct` version tag; `null` if the tag is absent.

7. `/model-info` reads the Production stage version only. Versions in other stages (e.g., rejected challengers) are ignored.

### `GET /model-info` — no Production model

8. Given no Production model version exists in the registry, when `GET /model-info` is called, the response is HTTP 200 with:

   ```json
   {
     "status": "no_production_model",
     "model_version": null,
     "stage": null,
     "training_date": null,
     "model_test_wape": null,
     "wape_lift_pct": null
   }
   ```

### `/heatmap` — simulated flag added

9. When `GET /heatmap` returns HTTP 200, the response envelope contains `"simulated": true` at the top level alongside the existing `generated_at`, `hour`, `zone_count`, and `predictions` fields.

10. All existing `/heatmap` acceptance criteria from prior PBIs remain satisfied: correct zone count, descending sort order, 503 on missing parquet, 404 on unknown hour, 400 on malformed hour parameter.

### Trainer — `wape_lift_pct` version tag

11. When the training pipeline promotes a challenger model to Production and `wape_lift_pct` is computable (not `None`), the trainer writes a `wape_lift_pct` version tag on the newly registered version using the same `client.set_model_version_tag` call pattern already used for `model_test_wape`.

12. When `wape_lift_pct` is `None` (no baseline overlap), the trainer does not write the `wape_lift_pct` tag. No error is raised.

### Tests

13. Three new test functions exist in `tests/test_serving.py`:
    - One covering `GET /health` happy path (parquet loaded, Production version present).
    - One covering `GET /health` degraded path (parquet loaded, no Production version).
    - One covering `GET /model-info` happy path (Production version present with both tags).
    All three mock `mlflow.tracking.MlflowClient` — no real registry is touched.

14. All pre-existing tests in `tests/test_serving.py` pass without modification.

---

## Edge Cases to Consider

- `max(bucket_time)` is in the future (e.g., predictions generated for a forecast window ahead of today): lag must clamp to `0`, not go negative.
- Predictions parquet is present but has zero rows: `last_prediction_bucket` and `data_publication_lag_days` should both be `null` rather than raising an exception.
- `creation_timestamp` from the MLflow registry is in milliseconds since epoch (not seconds); conversion to UTC ISO-8601 must divide by 1000 before passing to `datetime.fromtimestamp`.
- `wape_lift_pct` tag value is stored as a string in MLflow (consistent with `model_test_wape`); `/model-info` must cast to `float` before returning, or return `null` on cast failure.
- Registry SQLite file does not exist on a fresh checkout: the `MlflowClient` constructor may raise; `/health` must catch this and return the degraded shape rather than a 500.
- Concurrent requests to `/health` during a server restart where `_predictions_df` is being replaced: existing module-global pattern does not lock; document as known limitation, not a blocker for this PBI.
- `hour` column in the predictions parquet may be timezone-aware or timezone-naive depending on how the simulator wrote it; `max(bucket_time)` comparison with `today_utc` must normalise to UTC.

---

## Out of Scope

- Migration of response shapes to Pydantic models — hand-built dicts continue as per existing pattern.
- A real simulated-clock implementation that replays data relative to a configurable reference time — `simulated: true` is hardcoded for the MVP.
- Real-time data ingestion or a live telemetry feed — the simulated flag explicitly covers this gap.
- A model rollback endpoint or any write operation against the registry.
- A retraining trigger endpoint.
- Pagination or filtering on `/model-info` (e.g., listing all versions or all stages).
- Rewiring `Predictor.load_model()` to read from the registry at inference time (deferred to PBI 21).
- Per-request lag recalculation — `last_prediction_bucket` is set at startup from the loaded parquet and refreshed only on process restart.

---

## Dependencies

- **PBI 18** (shipped, commit `783f5ff`): provides `wape_lift_pct` as a logged MLflow run metric and the `compute_wape_lift` utility in `app/core/metrics.py`.
- **PBI 20** (in PR #9, this branch builds on top): establishes the MLflow registry, the Production stage, the `model_test_wape` version tag pattern, and the `client.set_model_version_tag` call in `app/training/trainer.py`.

---

## Open Questions

None. All product decisions have been confirmed by the team.
