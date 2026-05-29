# Technical Brief: PBI 22 — Prediction Log + Actuals-Join Job

## 1. Summary

Build a two-part monitoring foundation: (a) `app/monitoring/prediction_log.py` writes every `/heatmap` response to a partitioned parquet log immediately after the sort at line 210 of `app/api/main.py`, and (b) `app/monitoring/actuals_join.py` provides a CLI-runnable function that reads accumulated prediction logs, left-joins them against the Gold `hourly_demand_featured.parquet` on `(pulocationid, bucket_time == hour)`, and writes a single joined parquet per invocation. No HTTP surface changes; no new dependencies; no schema changes to Gold or the existing predictions parquet.

---

## 2. Data Model Changes

### Prediction Log Parquet

**Path pattern:** `{monitoring.predictions_path}/dt={YYYY-MM-DD}/served_at={ISO-ts}-{uuid8}.parquet`
- `dt=` is the UTC calendar date of the write (`datetime.now(tz=timezone.utc).date().isoformat()`).
- `ISO-ts` is the UTC instant with `:` replaced by `-` and `.` replaced by `-` so it is filename-safe (e.g., `2024-11-15T23-59-59-123456Z`).
- `uuid8` is `uuid.uuid4().hex[:8]`.
- Default root from `config.yml`: `data/monitoring/predictions`.
- Directory created with `os.makedirs(..., exist_ok=True)` before first write.

| Column | pandas dtype | Nullable |
|---|---|---|
| `bucket_time` | `datetime64[us, UTC]` | No |
| `pulocationid` | `int64` | No |
| `predicted` | `float64` | No |
| `model_version` | `object` (string) | Yes |
| `served_from_stage` | `object` (string) | Yes |
| `served_at` | `datetime64[us, UTC]` | No |

Source mapping from `/heatmap` response dict:
- `bucket_time` ← `pd.Timestamp(item["bucket_time"], tz="UTC")` (parse ISO string per edge-case note in story)
- `predicted` ← `item["predicted_trip_count"]`
- `model_version` ← `response["model_version"]` (same value for all rows in one call)
- `served_from_stage` ← `response["served_from_stage"]`
- `served_at` ← `datetime.now(tz=timezone.utc)` set inside `log_predictions`, not passed by caller

### Joined Output Parquet

**Path pattern:** `{monitoring.joined_path}/dt={start}_to_{end}.parquet`
- `start` and `end` are `YYYY-MM-DD` strings from CLI args.
- Default root from `config.yml`: `data/monitoring/joined`.

| Column | pandas dtype | Nullable |
|---|---|---|
| `bucket_time` | `datetime64[us, UTC]` | No |
| `pulocationid` | `int64` | No |
| `predicted` | `float64` | No |
| `actual` | `float64` | Yes (NaN where Gold has no match) |
| `model_version` | `object` (string) | Yes |
| `served_from_stage` | `object` (string) | Yes |
| `served_at` | `datetime64[us, UTC]` | No |

Join key: `pulocationid` (int64) and `bucket_time == hour` (timestamp). Gold's `hour` column is a naive `pd.Timestamp`; cast it to UTC-aware with `pd.to_datetime(gold_df["hour"]).dt.tz_localize("UTC")` before joining to match `bucket_time`.

### Config Addition

Add to `config.yml` after the `serving:` block:

```yaml
monitoring:
  predictions_path: data/monitoring/predictions
  joined_path: data/monitoring/joined
```

The API reads it as `config["monitoring"]["predictions_path"]` in `get_heatmap`.

---

## 3. Background Flow / Process Flow

### A. Prediction log write (per `/heatmap` request)

1. `get_heatmap` runs normally — guards 503/400/404 still fire before this path.
2. Sort at line 207–210 of `app/api/main.py` completes.
3. Response dict assembled at lines 212–220 (existing code, unchanged).
4. **New hook inserted between line 210 and line 212** (between the `predictions.sort(...)` call and the `return {…}` statement):

```python
try:
    from app.monitoring import prediction_log
    prediction_log.log_predictions(
        response, config["monitoring"]["predictions_path"]
    )
except Exception as exc:  # noqa: BLE001
    logger.warning("Failed to write prediction log: %s", exc)
```

5. `log_predictions(response, path)` runs synchronously (~1–2 ms for 263 rows). Safe in an async handler because it is pure CPU + local disk I/O with no blocking network call.
6. Handler returns HTTP 200 regardless of log outcome.

> Note: `config` must be in scope. Currently `config = load_config()` is called in `lifespan()` only. The handler must either store config in a module-level global (e.g., `_config: dict`) set during lifespan, or call `load_config()` at the top of the hook block. **Recommended:** add `_config: dict[str, Any] = {}` module global set in `lifespan()` alongside `_predictions_df`, then reference `_config["monitoring"]["predictions_path"]` in the hook.

### B. `log_predictions` internal sequence

1. If `response["predictions"]` is empty: `logger.debug("Skipped empty predictions")`; return `None`.
2. Compute `served_at = datetime.now(tz=timezone.utc)`.
3. Build list of row dicts from `response["predictions"]`; rename `predicted_trip_count` → `predicted`; parse `bucket_time` string to UTC timestamp; attach `model_version`, `served_from_stage`, `served_at`.
4. Build `pd.DataFrame`; cast dtypes explicitly (`pulocationid` → int64, timestamps → `datetime64[us, UTC]`).
5. Derive `dt_str = served_at.date().isoformat()`.
6. Build filename: `served_at_{ts_safe}-{uuid8}.parquet` where `ts_safe = served_at.strftime("%Y-%m-%dT%H-%M-%S-%fZ")`.
7. `os.makedirs(out_dir, exist_ok=True)`.
8. `df.to_parquet(out_path, index=False)`.
9. Return `Path(out_path)`.

### C. `join_actuals` CLI sequence

1. CLI parses `--start-date` and `--end-date` via `argparse`; calls `load_config()`.
2. For each date in `[start_date, end_date]` inclusive: glob `{predictions_root}/dt={date}/*.parquet`. Skip date silently if directory absent or empty.
3. Concatenate all found parquets into one DataFrame with `pd.concat(..., ignore_index=True)`.
4. If no files found at all: exit with informative message (zero-row join would be misleading).
5. `pd.read_parquet({gold_path}/hourly_demand_featured.parquet)` — raise `FileNotFoundError` with clear message if file absent (do not swallow).
6. Cast Gold `hour` column to UTC-aware: `pd.to_datetime(gold["hour"]).dt.tz_localize("UTC")`.
7. Left-merge predictions on `["pulocationid", "bucket_time"]` against Gold on `["pulocationid", "hour"]`; keep only `trip_count` from Gold; rename to `actual`.
8. Drop all Gold feature columns except `actual`.
9. `os.makedirs({output_root}/joined, exist_ok=True)`.
10. Write to `{output_root}/joined/dt={start}_to_{end}.parquet`; return `Path`.

---

## 4. API Changes

**File:** `app/api/main.py`

Changes are minimal and additive only.

1. **Add module-level global** (after line 39):
   ```python
   _config: dict[str, Any] = {}
   ```

2. **Populate in `lifespan()`** (add to the `global` declaration line and assign `_config = config` after `load_config()` call at line 90):
   ```python
   global _predictions_df, ..., _config
   _config = load_config()
   ```

3. **Insert hook in `get_heatmap`** between `predictions.sort(...)` (line 210) and `return {` (line 212):
   ```python
   try:
       from app.monitoring import prediction_log
       prediction_log.log_predictions(
           {  # build slim response for log before full dict is assembled
               "predictions": predictions,
               "model_version": _model_version,
               "served_from_stage": _served_from_stage,
           },
           _config.get("monitoring", {}).get("predictions_path", "data/monitoring/predictions"),
       )
   except Exception as exc:  # noqa: BLE001
       logger.warning("Failed to write prediction log: %s", exc)
   ```
   The hook passes a minimal dict matching what `log_predictions` needs; the full `return {}` block at 212 is unchanged.

4. **No new HTTP endpoints.** No changes to request/response shapes. All existing `/heatmap` behaviour (sort order, 503/404/400 codes) is unaffected.

**Tenant / isolation:** This feature is global — it logs all API responses into a shared `data/monitoring/` directory. There is no per-user or per-mailbox isolation boundary; the pipeline is single-tenant.

---

## 5. Frontend Changes

None.

---

## 6. Tests Required

**File:** `tests/test_monitoring.py`

Fixture approach: `tmp_path` for all I/O; inline synthetic DataFrames; no real `data/` reads; patch `datetime.now` where timestamp determinism is needed (`unittest.mock.patch("app.monitoring.prediction_log.datetime")`).

### TestPredictionLog

**Happy path**

- `test_log_predictions_valid_response_writes_file` — Arrange 3-zone response dict; Act call `log_predictions(response, str(tmp_path))`; Assert returned `Path` exists and `pd.read_parquet(path)` succeeds. _(Covers AC 1, 2)_

- `test_log_predictions_valid_response_schema_columns` — Same call; Assert columns == `{"bucket_time", "pulocationid", "predicted", "model_version", "served_from_stage", "served_at"}`. _(Covers AC 3)_

- `test_log_predictions_two_calls_same_day_separate_files` — Patch `datetime.now` to return same UTC day; call twice; Assert two distinct files exist in the same `dt=` directory, both non-zero bytes. _(Covers AC 4)_

**Failure / edge cases**

- `test_log_predictions_empty_predictions_returns_none` — Pass `response["predictions"] = []`; Assert return value is `None` and no parquet file written under `tmp_path`. _(Covers AC 5, resolved decision 1)_

- `test_log_predictions_null_model_version_writes_null` — Pass `model_version=None`, `served_from_stage=None`; Assert written parquet has null values in those columns without raising. _(Covers edge case: nullable fields)_

- `test_log_predictions_bucket_time_is_utc_timestamp` — Assert `bucket_time` dtype in written parquet is `datetime64[us, UTC]`, not object/string. _(Covers edge case: ISO string must be parsed)_

### TestActualsJoin

**Happy path**

- `test_join_actuals_row_count_matches_predictions` — Write synthetic prediction parquets for two zones, two buckets; call `join_actuals`; Assert joined row count == total prediction rows. _(Covers AC 10, 11, 12)_

- `test_join_actuals_actual_non_null_where_gold_matches` — Gold fixture includes matching `(pulocationid, hour)` row; Assert `actual` is non-null for that row. _(Covers AC 11)_

- `test_join_actuals_actual_null_where_gold_gap` — Gold omits one zone; Assert `actual` is `NaN` for that zone's rows. _(Covers AC 11)_

**Failure / edge cases**

- `test_join_actuals_missing_gold_raises_file_not_found` — Gold parquet absent; Assert `FileNotFoundError` is raised with a non-empty message. _(Covers edge case from story)_

- `test_join_actuals_missing_date_dir_silently_skipped` — Prediction dir for one date in range is absent; Assert no exception and joined output contains only rows from the date that has files. _(Covers AC 10, edge case)_

- `test_join_actuals_output_path_matches_pattern` — Assert returned `Path` filename equals `dt={start}_to_{end}.parquet`. _(Covers AC 12, resolved decision 3)_

### TestAPIHook (in `tests/test_serving.py` or `tests/test_monitoring.py`)

- `test_heatmap_hook_called_on_valid_response` — Use existing `client` fixture; patch `app.monitoring.prediction_log.log_predictions` as a MagicMock; GET `/heatmap`; Assert mock was called once with a dict containing `"predictions"` key. _(Covers AC 6)_

- `test_heatmap_returns_200_when_log_raises` — Patch `log_predictions` to raise `OSError`; GET `/heatmap`; Assert HTTP 200 and response body contains `predictions`. _(Covers AC 7, resolved decision 2)_

---

## 7. Risks and Open Questions

- **`_config` global scope in `get_heatmap`:** The handler currently has no reference to `config` at request-time (it was only in `lifespan()`). Adding `_config` as a module global is the simplest fix and consistent with the existing `_predictions_df` / `_zone_lookup` pattern. The builder should verify no race condition between lifespan startup and early requests; FastAPI's lifespan context ensures `yield` completes before the server accepts requests, so this is safe.

- **Gold `hour` timezone naive vs. log `bucket_time` UTC-aware:** Gold timestamps are stored naive (America/New_York wall-clock but stored without tzinfo). The brief specifies `tz_localize("UTC")` for the join, which is technically incorrect if the data was collected in NYC local time. However, the prediction log writes `bucket_time` parsed from the `/heatmap` response's `bucket_time` field, which itself is derived from `target_hour.isoformat()` (a naive pandas Timestamp from the predictions parquet). Both sides of the join will therefore be "naive stored as UTC" in the same way, so the join key is consistent — but the absolute UTC wall-clock values are wrong by ~4–5 hours for NYC data. This is a pre-existing issue, not introduced by PBI 22. Surface to the team as a known debt for PBI 23.

- **`bucket_time` ISO string format from `/heatmap`:** `target_hour.isoformat()` on a naive `pd.Timestamp` produces a string without timezone offset (e.g., `"2021-03-01T00:00:00"`). `pd.Timestamp(s, tz="UTC")` will correctly localize this to UTC. Confirm the parsing works for the 15-min bucket offset format (e.g., `"2021-03-01T00:15:00"`) — it will.

- **Performance of `pd.concat` over large date ranges:** For a 30-day window at 96 calls/day and 263 zones/call, `pd.concat` loads ~760k rows into memory before writing. This is well within normal pandas limits, but operators running year-long backfills should be warned to invoke the CLI in monthly chunks.

- **`from app.monitoring import prediction_log` inside handler:** The import is inside the try/except to avoid a hard startup failure if the module has a syntax error. This is intentional defensive coding, but means the import cost (~0.1 ms, cached after first call) is paid on the first successful request. Alternative is to hoist the import to module-level and accept that a broken monitoring module prevents API startup. **Recommendation:** move import to module-level (standard Python practice); a broken monitoring module should be a deployment blocker, not silently swallowed.

---

## 8. Files That Will Change

**Backend — new**
- `(new) app/monitoring/prediction_log.py` — `log_predictions(response, path) -> Path | None`
- `(new) app/monitoring/actuals_join.py` — `join_actuals(...)` function + `argparse` CLI entry point (`if __name__ == "__main__":`)

**Backend — modified**
- `app/api/main.py` — add `_config` module global; populate in `lifespan()`; insert try/except hook between lines 210 and 212 in `get_heatmap`
- `config.yml` — add `monitoring:` top-level section with `predictions_path` and `joined_path` keys

**Tests — new**
- `(new) tests/test_monitoring.py` — `TestPredictionLog` (6 tests) + `TestActualsJoin` (6 tests)

**Tests — modified**
- `tests/test_serving.py` — add `test_heatmap_hook_called_on_valid_response` and `test_heatmap_returns_200_when_log_raises` (2 tests); patch `app.monitoring.prediction_log.log_predictions` via `unittest.mock.patch`
