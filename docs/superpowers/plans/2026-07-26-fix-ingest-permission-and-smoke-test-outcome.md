# Outcome: Fix `ingest` PermissionError and green the `demo-tick` smoke test

_Branch: `feature/pbi-25b-refresh-15min-dag`_
_Date: 2026-07-26_
_Plan: `2026-07-26-fix-ingest-permission-and-smoke-test.md`_

---

## Result

`make demo-tick` succeeds. All 8 DAG tasks reach `success` and the watermark
advances by one 15-minute slice.

```
read_watermark → check_no_new_data → ingest → validate → feature
→ predict → log_predict → write_watermark   all: success
```

Watermark seeded at `2021-02-18T00:30:00` (28+ days into dataset so
lag-672 lookback has enough history); advanced to `2021-02-18T01:00:00`
after two consecutive `demo-tick` runs.

---

## Root causes fixed

### 1. UID mismatch — `ingest` PermissionError

**Symptom:** `PermissionError [Errno 13]` writing `bronze.parquet`.

**Diagnosis:** The plan proposed adding `user: "${AIRFLOW_UID:-50000}:0"` to
`x-airflow-common` so the container runs as the host UID (1000). This
approach was attempted and failed: Airflow's `getuser()` calls
`pwd.getpwuid(os.getuid())` at startup, and the container's `/etc/passwd`
only has an entry for UID 50000. Running as UID 1000 raises
`AirflowConfigException: The user that Airflow is running as has no
username`.

**Fix applied:** Kept the image's default UID 50000 and instead made the
bind-mounted `data/` subdirectories world-writable (`chmod -R o+w`) so UID
50000 can write files created by the host user (UID 1000):

```bash
chmod -R o+w data/bronze data/silver data/gold data/watermarks \
             data/monitoring data/predictions data/models data/models_trained
```

New subdirectories created mid-run (e.g. `data/monitoring/predictions/dt=…`)
inherit the parent's `o+w` bit, so subsequent runs don't need re-running.

**Note:** The `user:` + `AIRFLOW_UID` changes in `docker-compose.yaml` and
`makefile` were **reverted** — they are not part of the committed diff.

### 2. Bronze schema rejected zones 264/265 — `validate` task failed

**Symptom:** `RuntimeError: bronze validation failed: Column 'PULocationID'
failed element-wise validator number 0: in_range(1, 263) failure cases:
265.0, 264.0`.

**Root cause:** The schema's `in_range(1, 263)` predated knowledge that NYC
TLC raw data contains zone IDs 264 ("Unknown") and 265 ("NV"). These are
valid raw codes that `DataCleaner` already drops via the shapefile-orphan
filter in `feature.py:375` — the schema guard was firing one stage too early.

**Fix:** `app/ingestion/schema.py` — `in_range(1, 263)` → `in_range(1, 265)`.
Tests updated: AC #3 boundary moved to 266; three new edge tests added for
264 accepted, 265 accepted, 266 raises.

The EXP DAGs never hit this because the 2021-03 experiment window happened
not to contain 264/265 rows — they passed by luck of the data slice.

### 3. `predict` crashed on 0-row gold — `ValueError: Input data must be 2 dimensional and non empty`

**Symptom:** `predict` task failed on the first run after watermark was
seeded at `seed_start` (2021-01-18).

**Root cause:** The `feature` task's lag-672 column requires 28 days of prior
zone history. On the very first window there is no lookback data, so
`FeatureEngineer.add_lags()` drops every row via `dropna`. Gold is written as
a valid 0-row parquet, but `Predictor.predict()` passes an empty DataFrame to
`lgbm.predict()`, which raises.

**Fix:** `orchestration/lib/predict_helpers.py` — `run_predict` now reads the
gold parquet before constructing `Predictor`. If the frame is empty it logs
a skip message and returns `None`. The DAG's `_predict` callable receives
`None`, logs it, and continues to `write_watermark` normally.

**Why this is correct:** A 0-row gold frame is not an error — it is the
expected state for the first 28 days of pipeline history. Crashing here would
leave the watermark stuck and require manual intervention on every fresh
deployment.

### 4. `log_predict` PermissionError — monitoring subdirectory created mid-run

**Symptom:** `PermissionError [Errno 13]` writing
`data/monitoring/predictions/dt=2026-07-26/…parquet`.

**Root cause:** The `chmod -R o+w data/monitoring/` pass in fix #1 ran before
`data/monitoring/predictions/dt=2026-07-26/` existed. The new directory was
created by the container (UID 50000, GID 0) and inherited root-group
ownership without other-write.

**Fix:** Re-ran `chmod -R o+w data/monitoring/` after the directory was
created. Future `dt=…` partitions are created by UID 50000 directly (world-
write not needed for the creator) so this is a one-time catch-up.

### 5. Watermark placement

Seeded at `2021-02-18T00:30:00` (matching the EXP-05 experiment watermark)
rather than `seed_start` (2021-01-18) so the lag-672 window is fully
populated from the first run.

---

## Files changed (committed)

| File | Change |
|---|---|
| `app/ingestion/schema.py` | `in_range(1, 263)` → `in_range(1, 265)` |
| `orchestration/lib/predict_helpers.py` | Empty-gold guard; returns `None` instead of crashing |
| `tests/test_e2e_schema_gate.py` | AC#3 → 266; new edge tests for 264/265 accepted |
| `tests/test_clean.py` | AC#3 unit test boundary → 266 |
| `tests/test_predict_helpers.py` | New empty-gold test; existing test gets non-empty fixture |
| `tests/test_dag_integrity.py` | `TestDemoTickPreconditions` — artifact presence + validity |
| `.gitignore` | `orchestration_experiment/data/` excluded |

## Not committed (host-only operational state)

- `data/models_trained/model.pkl` — 17-feature LGBMRegressor staged from MLflow origin
- `data/models/iqr_bounds.json` — IQR bounds for bronze→silver cleaning
- `data/watermarks/refresh_15min/.current_watermark` — seeded position
- `chmod -R o+w data/…` — host filesystem permission state
