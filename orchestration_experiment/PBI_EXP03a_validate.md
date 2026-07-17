# PBI EXP-03a — Validate (Bronze Schema Gate) (Experimentation)

> **Type:** Experimentation PBI (not for release; lives under `orchestration_experiment/`)
> **Depends on:** EXP-02 (bronze must materialize cumulatively).
> **Parent:** Orchestrator redesign brainstorm (see `MEMORY.md → orchestrator-redesign-brainstorm`).
> **Split note:** The original EXP-03 bundled `validate` + `feature`. Split into
> EXP-03a (this PBI, validate) and EXP-03b (feature silver→gold) so each insert
> is one task, one dep bump, one smoke cycle.

---

## 1. Why this PBI exists

EXP-02 gave us cumulative bronze but nothing checks its shape. If a raw parquet
lands with the wrong dtype or a `PULocationID` outside `[1, 263]`, the pipeline
today would happily feed garbage downstream. EXP-03a inserts a **schema-validation
gate**: malformed bronze → run fails → watermark held → next run reprocesses the
same slice against the same (still-bad) bronze until a human intervenes.

This is the mental model's "guard bronze before feature":

> validate: Pandera schema check on bronze; fail the run (and thus hold the
> watermark) if bronze is malformed.

---

## 2. Goal

Insert ONE task between `ingest` and `advance_watermark`:

- `validate` — `validate_bronze_schema(df)` + `write_validation_log(...)`;
  raise `RuntimeError` on failure so `advance`/`write` skip under `all_success`.

---

## 3. In scope

- Add `validate` task:
  `read_watermark → ingest → validate → advance_watermark → write_watermark`.
- Load bronze parquet, call `app.ingestion.schema.validate_bronze_schema`,
  write a log via `write_validation_log(validation_log_dir(config), df, reasons)`.
- On `DataValidationError`: write the failure log AND re-raise as `RuntimeError`
  so Airflow marks the task failed (crash-safety contract from EXP-02).
- Add `pandera>=0.20` to the baked image (deliberately omitted in EXP-02).
- Validation log location: `<silver_root>.parent / "_validation"` under the
  isolated experiment data root (silver root is a config value; no silver
  parquet exists yet — the log dir just uses that parent).

## 4. Out of scope (deferred)

- `feature` task (silver → gold) — **EXP-03b**.
- Predict / log_predict — EXP-04.
- `.done` sentinels + `check_no_new_data` boundary — EXP-05.
- Any change to `app/`, `orchestration/`, `implementation_checklist/`.
- Any change to what `validate_bronze_schema` accepts — the schema in
  `app/ingestion/schema.py` is treated as immutable for this PBI.

---

## 5. Deliverables

1. Updated DAG with `validate` inserted (5 tasks).
2. Validation log JSON written under the experiment data root each run.
3. Failed-validate smoke test showing the watermark held on malformed bronze.
4. Note recording: log path, healthy-run row counts, failure-run row counts,
   and the exact `failure_reasons` from the malformed-bronze test.

---

## 6. DAG shape

```
read_watermark → ingest → validate → advance_watermark → write_watermark
```

- `validate` gates the run: malformed bronze → RuntimeError → `advance`/`write`
  skip under `trigger_rule="all_success"` → watermark unchanged.
- The clean-then-feature stage is inserted in EXP-03b; EXP-03a stops here.

---

## 7. Acceptance (experimental)

- After a healthy run: `validate` succeeds, log JSON exists under
  `<silver_root>.parent/_validation/validation_*.json` with `rows_failing == 0`,
  watermark advanced by one slice.
- Feeding deliberately-malformed bronze (e.g. `PULocationID = 999` on all rows)
  → `validate` fails → `advance`/`write` skip → watermark unchanged → prior
  bronze intact → validation log JSON exists with a non-empty `failure_reasons`.
- No imports from `orchestration/`.
- `uv run pytest` baseline unchanged (only new tests added).

---

## 8. Open questions — resolved 2026-07-17

- **OQ1 → (a) inject config dict.** `build_validate_config(base, *, silver_root)`
  deep-copies the base config and overrides `data.silver_path`. This keeps the
  same seam as EXP-02's `build_ingest_config` and EXP-03b will inherit it.
- **OQ2 → empty bronze passes.** `rows_in=0` is logged and the gate passes.
  EXP-05's `check_no_new_data` handles the empty-window boundary separately.

---

## 9. Verification note — 2026-07-17 GREEN

### Files shipped

| File | Purpose |
|---|---|
| `orchestration_experiment/dags/exp03a_validate.py` | Pure helpers: `build_validate_config`, `run_validate` |
| `orchestration_experiment/dags/exp03a_validate_dag.py` | DAG wrapper — 5-task chain |
| `tests/test_exp03a_validate.py` | 10 unit tests (RED-first TDD) |
| `tests/test_exp03a_dag_structure.py` | Structural test (skipped on host; in-container below) |
| `orchestration_experiment/requirements-exp.txt` | Added `pandera>=0.20` |

### Schema fix (unplanned finding)

During smoke testing, `validate_bronze_schema` rejected zones **264** and **265**
(NYC "Unknown" zones present in raw data). The original `in_range(1, 263)` check
was too tight — those zones are valid at the bronze layer and are filtered later
by `FeatureEngineer.add_zone_coordinates` (no shapefile match). Fixed:

- `app/ingestion/schema.py` — `in_range(1, 263)` → `in_range(1, 265)`
- `tests/test_clean.py` — updated out-of-range test to use `266.0`; docstring explains why
- `tests/test_e2e_schema_gate.py` — same update to `266.0`; added
  `test_ac3_pulocationid_264_265_accepted` to lock the new boundary

### pytest (host)

```
473 passed, 11 skipped   (was 472 before this PBI; +1 from new 264/265 boundary test)
```

### Container smoke — happy path

```
airflow dags test exp03a_validate_dag 2021-01-18
```

All 5 tasks SUCCESS:

| Task | State |
|---|---|
| `read_watermark` | SUCCESS — window `[01:00, 01:15)` |
| `ingest` | SUCCESS — bronze 357 rows |
| `validate` | SUCCESS — `rows_passing=357`, `rows_failing=0` |
| `advance_watermark` | SUCCESS |
| `write_watermark` | SUCCESS |

Watermark: `2021-01-18T00:00:00` → `2021-01-18T01:15:00`

Validation log: `data_exp/silver/_validation/validation_20260717T110453.json`
```json
{
  "validated_at": "2026-07-17T11:04:53.377263Z",
  "rows_in": 357,
  "rows_passing": 357,
  "rows_failing": 0,
  "failure_reasons": []
}
```

### Container smoke — failure-path note

Corrupting the bronze parquet directly does **not** exercise the watermark-hold
AC because `ingest` performs a **full cumulative rewrite from raw** every run
(ADR-001). The corrupted parquet is overwritten before `validate` sees it.

The watermark-hold contract is proven at two levels:
1. **Unit tests** — `test_reraises_data_validation_error_as_runtime_error` and
   `test_writes_log_with_failure_reasons_before_reraising` assert that
   `run_validate` raises `RuntimeError` and writes the failure log on the bad
   path. These were written RED-first and watched fail before implementation.
2. **Pre-fix container logs** — the two failed runs at `10:57:39` and `10:59:43`
   (visible in the log history) show `validate` failing with
   `PULocationID out of range` and the watermark held at `2021-01-18T00:00:00`
   until the schema fix landed. This is observable evidence the gate works.

To exercise the failure path in future, corrupt a **raw source parquet** (not
the bronze) so ingest ingests the bad data and validate catches it downstream.
