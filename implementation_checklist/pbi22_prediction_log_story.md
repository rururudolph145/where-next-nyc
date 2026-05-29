# PBI 22 — Prediction Log + Actuals-Join Job

## Why This Matters

Today the API serves predictions with no durable record of what was returned, to whom, or when. When Gold actuals eventually arrive (typically 1–2 months after the prediction window), there is no dataset to join against — making it impossible to compute any live accuracy metric or detect model drift. This PBI creates the two-part monitoring foundation: a lightweight per-request log that captures every served prediction at the time it is served, and an offline join job that enriches that log with Gold actuals once they exist. PBI 23 (drift detection) depends entirely on the joined output produced here.

---

## User Story

As a pipeline operator, I want every `/heatmap` response to be durably logged in partitioned parquet files and joined against eventual Gold actuals by a scheduled job, so that I have a complete, queryable dataset of predictions paired with ground-truth demand for drift monitoring and model evaluation.

---

## Acceptance Criteria

### Prediction log writer (`app/monitoring/prediction_log.py`)

1. A function `log_predictions(response: dict[str, Any], path: str) -> Path` exists and is importable from `app.monitoring.prediction_log`.

2. When called with a valid `/heatmap` response dict and a monitoring root path, `log_predictions` writes exactly one parquet file with path pattern:
   `{root}/predictions/dt={YYYY-MM-DD}/served_at={ISO-timestamp}-{uuid8}.parquet`
   where `dt` is the UTC calendar date of the write, `ISO-timestamp` is the full UTC instant (safe for filenames), and `uuid8` is the first 8 characters of a UUID4. The function returns the `Path` of the written file.

3. The written parquet contains exactly one row per prediction in `response["predictions"]` (263 rows for a full `/heatmap` response). The schema is:

   | Column | Type |
   |---|---|
   | `bucket_time` | timestamp (UTC) |
   | `pulocationid` | int64 |
   | `predicted` | float64 |
   | `model_version` | string (nullable) |
   | `served_from_stage` | string (nullable) |
   | `served_at` | timestamp (UTC) |

   `bucket_time` is populated from each prediction's `bucket_time` field in the response (the 15-min-bucketed prediction window). `served_at` is set inside `log_predictions` using `datetime.now(tz=timezone.utc)` — the caller does not pass it. There is no `actual` column in the prediction log.

4. Two calls to `log_predictions` within the same UTC calendar day both write to the same `dt=YYYY-MM-DD/` directory but produce two separate files (distinct `uuid8` suffixes). No existing file is overwritten. Concurrent writes from multiple workers cannot corrupt each other because each request produces a uniquely named file.

5. Given `response["predictions"]` is an empty list, `log_predictions` skips the file write entirely, emits a `logging.debug` message, and returns `None`. No zero-row parquet is written. _(See Open Questions — this behaviour is a default pending confirmation.)_

### API hook (`app/api/main.py`)

6. After computing a `/heatmap` response dict but before returning it to the caller, the handler calls `prediction_log.log_predictions(response, config["monitoring"]["predictions_path"])`.

7. If `log_predictions` raises any exception (permission error, disk full, etc.), the handler catches it, emits a `logging.warning` with the exception detail, and still returns the computed response with HTTP 200. A monitoring write failure must never degrade the serving path.

8. All existing `/heatmap` acceptance criteria remain satisfied: correct zone count, descending sort by `predicted_trip_count`, HTTP 503 on missing predictions parquet, HTTP 404 on unknown hour, HTTP 400 on malformed hour parameter.

### Actuals join script (`app/monitoring/actuals_join.py`)

9. A function `join_actuals(start_date: date, end_date: date, predictions_root: str, gold_path: str, output_root: str) -> Path` exists and is importable from `app.monitoring.actuals_join`. `end_date` is inclusive.

10. `join_actuals` reads all parquet files under `{predictions_root}/dt=YYYY-MM-DD/*.parquet` for every calendar date in `[start_date, end_date]`. Dates with no files in that directory are silently skipped (no files found = zero rows contributed for that date).

11. The function left-joins the collected prediction rows against the Gold parquet on `(pulocationid, bucket_time == hour)`, where `hour` is the Gold layer's bucketed timestamp column and `trip_count` becomes the `actual` column. Where Gold has no matching row for a given `(pulocationid, bucket_time)` pair, `actual` is `NaN` (null). No prediction rows are dropped.

12. The joined output is written to `{output_root}/joined/dt={start}_to_{end}.parquet` (one file per CLI invocation) with schema:

    | Column | Type |
    |---|---|
    | `bucket_time` | timestamp (UTC) |
    | `pulocationid` | int64 |
    | `predicted` | float64 |
    | `actual` | float64 (nullable) |
    | `model_version` | string (nullable) |
    | `served_from_stage` | string (nullable) |
    | `served_at` | timestamp (UTC) |

    The function returns the `Path` of the written file.

13. A CLI entry point using `argparse` accepts `--start-date YYYY-MM-DD` and `--end-date YYYY-MM-DD` (both required). All path arguments are read from `config.yml`. Running the script with valid dates and existing prediction parquets completes without error and writes the joined parquet.

### Config (`config.yml`)

14. `config.yml` gains a new top-level `monitoring:` section with at least two keys:
    - `predictions_path: data/monitoring/predictions`
    - `joined_path: data/monitoring/joined`

    Both keys are present in the default `config.yml` committed to the repository.

### Tests (`tests/test_monitoring.py`)

15. `TestPredictionLog` contains at least three test cases:
    - `test_happy_path_writes_file` — calls `log_predictions` with a synthetic 3-zone response; asserts the returned `Path` exists on disk and is readable via `pd.read_parquet`.
    - `test_schema_columns` — asserts the written parquet contains exactly the six columns listed in criterion 3, in any order.
    - `test_multiple_calls_same_day_produce_separate_files` — calls `log_predictions` twice with the same response; asserts two distinct files exist in the same `dt=` directory and neither is zero bytes.

    All tests use `tmp_path` and do not touch the real `data/` directory.

16. `TestActualsJoin` contains at least three test cases:
    - `test_row_count_matches_predictions` — synthetic prediction parquets covering two zones and two buckets; asserts the joined output has exactly the same number of rows as the total prediction rows read.
    - `test_actual_non_null_where_gold_matches` — Gold fixture contains a matching `(pulocationid, hour)` row; asserts `actual` is non-null for that row in the joined output.
    - `test_actual_null_where_gold_gap` — Gold fixture omits one zone; asserts `actual` is `NaN` for that zone's rows in the joined output.

    All tests use `tmp_path`, write synthetic parquets inline, and do not read from the real Gold or monitoring directories.

---

## Edge Cases to Consider

- A prediction's `bucket_time` string in the `/heatmap` response is an ISO 8601 string — `log_predictions` must parse it to a timezone-aware timestamp before writing to parquet, rather than storing a raw string.
- The `dt=YYYY-MM-DD/` partition directory may already exist from a previous run. Writing a new uniquely-named file into an existing directory must succeed without error.
- Midnight UTC boundary: a request served at 23:59:59 UTC and one at 00:00:01 UTC the next day must land in different `dt=` directories even if their `bucket_time` values fall in the same 15-min window.
- The actuals-join window spans a date range where some `dt=` directories exist and others do not (e.g., the API was down on one day). Missing directories are skipped silently; the joined output contains only the rows from dates that have prediction files.
- Gold parquet may not exist at all when the join job runs. `join_actuals` should raise a clear `FileNotFoundError` — not a silent empty join — so the operator knows Gold hasn't been published yet.
- A prediction window that falls in the future (simulated data) will never have a matching Gold row. All such rows will have `actual = NaN`. This is expected behaviour, not an error.
- The `model_version` and `served_from_stage` fields in the response envelope may be `None` (e.g., when running without PBI 21 active). The prediction log must accept nullable values and write them as null rather than raising.
- The output directory `{output_root}/joined/` may not exist when the join job runs for the first time. The script must create it with `os.makedirs(..., exist_ok=True)`.

---

## Out of Scope

- Drift detection, alerting, or any statistical analysis of the joined output — that is PBI 23.
- Pandera schema validation for prediction log or joined parquet files — deferred to PBI 23.
- Retention, cleanup, or expiry policies for files under `data/monitoring/` — deferred.
- Real Airflow DAG wiring — this PBI only requires the script to be CLI-runnable. PBI 25 handles DAG integration.
- Retry logic or back-off when the Gold parquet read fails transiently — a single attempt per CLI run is sufficient.
- Per-source-date output files from `actuals_join.py` (one file per `dt=YYYY-MM-DD` rather than one file per CLI run) — see Open Questions.
- Any schema changes to the `/heatmap` response itself — the log writer is a read-only consumer of the existing response shape.
- Changes to the Bronze, Silver, or Gold pipeline — this PBI only reads Gold, it does not modify how Gold is produced.
- Authentication or access control for the `data/monitoring/` directory.

---

## Dependencies

- **PBI 21** (strongly recommended before merge): adds `model_version` and `served_from_stage` to the `/heatmap` response envelope and to the predictions parquet. Without PBI 21, those fields will be null in the prediction log — which is acceptable per criterion 3 but reduces monitoring value. PBI 22 may be developed in parallel provided the null-value path is tested.
- **No other PBIs are blocking.** The `app/monitoring/__init__.py` placeholder already exists; `data/monitoring/` will be created on first write.

---

## Open Questions

1. **Empty-predictions response**: should `log_predictions` skip the write entirely when `response["predictions"]` is empty (current default in criterion 5), or write a zero-row parquet so PBI 23 can distinguish "API served zero predictions" from "API was never called"? Recommendation: skip and debug-log — a zero-row parquet adds operational noise with no analytic value at this stage.

2. **Log-write failure behaviour**: confirm that "emit a `logging.warning` and return the response anyway" (criterion 7) is the correct contract, rather than surfacing an HTTP 500. Recommendation: log and continue — monitoring infrastructure must never degrade the serving SLA.

3. **Joined output naming**: criterion 12 specifies one file per CLI invocation (`dt={start}_to_{end}.parquet`). An alternative is one file per source date (`dt=YYYY-MM-DD.parquet`) written inside the `joined/` directory, which would make PBI 23's rolling-window queries simpler. Which convention does the team prefer before implementation begins?
