# Technical Brief: PBI 27 — `/health`, `/model-info`, and `/heatmap` Simulated Flag

## 1. Summary

Add two new read-only diagnostic endpoints (`GET /health`, `GET /model-info`) to `app/api/main.py` that expose prediction freshness and model provenance, and add `"simulated": true` to the existing `/heatmap` envelope. The trainer's promotion path in `app/training/trainer.py` gains one additional `set_model_version_tag` call to write `wape_lift_pct` at promotion time so `/model-info` can surface it.

---

## 2. Data Model / Config Changes

**No `config.yml` changes required.** The MLflow tracking URI (`sqlite:///mlflow.db`) and model name (`mlflow.model_name`) already exist from PBI 20.

**New MLflow version tag — written at promotion time:**

| Tag key | Type (stored) | Written by | Condition |
|---|---|---|---|
| `wape_lift_pct` | string (float-parseable) | `app/training/trainer.py` | Only when `wape_lift_pct` is not `None` and not `float('inf')` |

The existing `model_test_wape` tag (written in the same block at lines 343–348 of `trainer.py`) is the sibling pattern to follow. Skip the tag write — do not write the string `"None"` or `"inf"` — when the value is non-finite.

---

## 3. Process Flow

**Trainer (promotion path):**

1. `apply_gate(...)` returns `"promoted"`.
2. Immediately after the existing `client.set_model_version_tag(..., "model_test_wape", ...)` call (trainer.py line 343), add a guarded sibling call: if `wape_lift_pct is not None` and `wape_lift_pct != float("inf")`, call `client.set_model_version_tag(..., "wape_lift_pct", str(wape_lift_pct))`.
3. No other trainer changes.

**`GET /health` request flow:**

1. If `_predictions_df is None`: raise `HTTPException(503)` — matches existing `/heatmap` pattern.
2. Compute `last_bucket = _predictions_df["hour"].max()` — returns `NaT` on empty frame.
3. If `_predictions_df` is empty: `last_prediction_bucket = null`, `data_publication_lag_days = null`.
4. Otherwise: normalise `last_bucket` to UTC (strip or assume tz-naive = America/New_York — see Risk c below); compute `lag = max(0, (today_utc.date() - last_bucket.date()).days)`.
5. Instantiate `MlflowClient()` in a try/except; on any exception return degraded shape.
6. Call `client.get_latest_versions(model_name, stages=["Production"])`; if empty list, `model_version = null`, `status = "degraded"`, else `model_version = versions[0].version`, `status = "ok"`.
7. Return assembled dict.

**`GET /model-info` request flow:**

1. Instantiate `MlflowClient()` in a try/except; on any exception or empty production list return `status = "no_production_model"` with all other fields null.
2. Extract `v = versions[0]`: `model_version = v.version`, `stage = v.current_stage`, `training_date = datetime.fromtimestamp(v.creation_timestamp / 1000, tz=timezone.utc).isoformat()`.
3. Cast tags: `model_test_wape = float(v.tags["model_test_wape"])` or null on missing/cast failure; same for `wape_lift_pct`.
4. Return assembled dict.

**`GET /heatmap` change:**

Add `"simulated": True` as a top-level key to the return dict at `app/api/main.py:197–202`. No logic change.

---

## 4. API Changes

All endpoints return `application/json`. No auth, no pagination, no rate limiting (consistent with existing pattern in `app/api/main.py`).

**`GET /health` — happy path (200):**
```json
{
  "status": "ok",
  "simulated": true,
  "data_publication_lag_days": 42,
  "last_prediction_bucket": "2024-03-15T23:45:00",
  "model_version": "3"
}
```

**`GET /health` — degraded (200, no Production version):**
```json
{
  "status": "degraded",
  "simulated": true,
  "data_publication_lag_days": 42,
  "last_prediction_bucket": "2024-03-15T23:45:00",
  "model_version": null
}
```

**`GET /health` — predictions unavailable (503):**
```json
{ "detail": "Predictions not available. Run the simulation pipeline first." }
```

**`GET /model-info` — happy path (200):**
```json
{
  "status": "ok",
  "model_version": "3",
  "stage": "Production",
  "training_date": "2024-03-10T08:22:01+00:00",
  "model_test_wape": 0.1823,
  "wape_lift_pct": 12.4
}
```

**`GET /model-info` — no Production model (200):**
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

**`GET /heatmap` envelope — updated (200):** Add `"simulated": true` alongside the existing `generated_at`, `hour`, `zone_count`, `predictions` fields.

---

## 5. Frontend Changes

None.

---

## 6. Tests Required

**Happy path**

- `test_get_health_happy_path_returns_ok_with_version` — `GET /health` with mocked `_predictions_df` (non-empty) and mocked `MlflowClient` returning a production version; asserts status 200, `status=="ok"`, `simulated==True`, `model_version` is a string, `data_publication_lag_days >= 0`.
- `test_get_model_info_happy_path_returns_all_fields` — `GET /model-info` with mocked `MlflowClient` returning a production version with both tags set; asserts status 200, `model_test_wape` and `wape_lift_pct` are floats, `training_date` is ISO-8601, `stage=="Production"`. (AC #6, #7)
- `test_get_heatmap_envelope_contains_simulated_flag` — `GET /heatmap` using existing `client` fixture; asserts `body["simulated"] == True`. (AC #9)
- `test_gate_writes_wape_lift_pct_tag_on_promotion` — in `TestRegistryGate`, mock `MlflowClient`, assert `set_model_version_tag` is called with key `"wape_lift_pct"` and a float-parseable string value when `wape_lift_pct` is not None/inf. (AC #11)

**Failure cases**

- `test_get_health_degraded_when_no_production_version` — `GET /health` with mocked `_predictions_df` (non-empty) and `MlflowClient.get_latest_versions` returning empty list; asserts status 200, `status=="degraded"`, `model_version` is null. (AC #4)
- `test_get_health_returns_503_when_predictions_not_loaded` — `_predictions_df = None`; asserts 503 with `detail`. (AC #5)
- `test_get_health_degraded_when_mlflow_client_raises` — `MlflowClient()` raises `Exception`; asserts 200 with `status=="degraded"`.
- `test_get_model_info_no_production_model_returns_null_fields` — `MlflowClient.get_latest_versions` returns empty; asserts 200, `status=="no_production_model"`, all other fields null. (AC #8)
- `test_gate_does_not_write_wape_lift_pct_tag_when_none` — `wape_lift_pct=None`; assert `set_model_version_tag` is NOT called with key `"wape_lift_pct"`. (AC #12)

**Edge cases**

- `test_get_health_lag_is_null_when_predictions_empty` — `_predictions_df` is an empty DataFrame; asserts `data_publication_lag_days` is null and `last_prediction_bucket` is null. (AC #3)
- `test_get_health_lag_is_zero_when_bucket_in_future` — `max(hour)` set to a future date; asserts `data_publication_lag_days == 0`, never negative. (AC #2)
- `test_get_model_info_returns_null_wape_when_tag_absent` — Production version exists but has no `model_test_wape` tag; asserts `model_test_wape` is null and no exception raised. (AC #6 null branch)
- `test_gate_does_not_write_wape_lift_pct_tag_when_inf` — `wape_lift_pct=float("inf")`; assert no tag written.

All new tests in `tests/test_serving.py` patch `mlflow.tracking.MlflowClient` via `unittest.mock.patch("app.api.main.MlflowClient")`. The trainer tag test extends `TestRegistryGate` in `tests/test_train.py`. All follow AAA pattern and naming convention `test_<function>_<condition>_<expected_result>`.

---

## 7. Risks and Open Questions

- **(a) SQLite lock / missing `mlflow.db` at request time.** Decided: catch any exception from `MlflowClient()` or `get_latest_versions()` and return the degraded / `no_production_model` shape. Neither endpoint raises 5xx for this case. The missing-db case is the most common scenario on a fresh checkout.

- **(b) Timezone of `last_prediction_bucket`.** The `hour` column in the predictions parquet is stored as a naive pandas Timestamp (America/New_York wall-clock, no `tzinfo`). The lag computation compares against `datetime.now(tz=timezone.utc).date()`. This produces correct results today because the parquet holds historical data well in the past; however, a naive timestamp from Dec 31 23:00 ET looks like Jan 1 04:00 UTC — the lag could be off by one day near midnight. For the MVP this is acceptable; document as a known limitation. The `last_prediction_bucket` string in the response is the raw naive ISO string (no `+00:00` suffix), consistent with existing `/heatmap` `hour` field behaviour.

- **(c) `MlflowClient` per-request vs. cached on app state.** The current module-global pattern (`_predictions_df`, `_zone_lookup`) does not include an MLflow client. Instantiating `MlflowClient()` per request opens a SQLite connection on each call. For the MVP request rate this is acceptable; if `/health` polling is frequent, a cached client on `app.state` should be considered. No infrastructure change needed now — flag for follow-up.

- **(d) Pre-existing `/heatmap` tests assert the envelope shape without `"simulated"`.** `test_heatmap_contains_required_fields` and `test_get_heatmap_envelope_fields_unchanged` do not assert `"simulated"` is absent, so adding the field will not break them. Builder must verify no test uses `assert body.keys() == {...}` strict equality — a scan of `tests/test_serving.py` shows none do.

- **(e) `wape_lift_pct` tag write location in `trainer.py`.** The tag must be written **after** `apply_gate()` returns `"promoted"` and `transition_model_version_stage` has been called, so the tag lands on the now-Production version. The current code structure (lines 341–380) calls `apply_gate` inside the `if wape_lift_pct is not None` branch — the `wape_lift_pct` tag write belongs immediately after `apply_gate` returns `"promoted"`, guarded by `if gate_result == "promoted"`. This is distinct from the `model_test_wape` tag which is written unconditionally before `apply_gate`.

---

## 8. Files That Will Change

**Backend**
- `app/api/main.py` — add `GET /health` endpoint, add `GET /model-info` endpoint, add `"simulated": True` to `/heatmap` return dict; add `MlflowClient` import from `mlflow.tracking`

**Training**
- `app/training/trainer.py` — add guarded `client.set_model_version_tag(..., "wape_lift_pct", ...)` call after gate promotion, inside the `if gate_result == "promoted":` block (lines ~379–381)

**Tests**
- `tests/test_serving.py` — add `TestHealthEndpoint` class (3 new tests covering AC #2–#5), add `TestModelInfoEndpoint` class (2 new tests covering AC #6–#8), add simulated-flag assertion to `TestHeatmapAcceptance` (AC #9)
- `tests/test_train.py` — add 2 new tests to `TestRegistryGate`: `test_gate_writes_wape_lift_pct_tag_on_promotion`, `test_gate_does_not_write_wape_lift_pct_tag_when_none`
