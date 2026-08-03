# PBI-MON-01 — Daily Actuals Backfill DAG

**Status:** Deferred — pending Monitoring ADR

## Problem

The 15-minute inference DAG writes `actual_trip_count = NaN` in the prediction
history (`data/predictions/history/dt=*.parquet`).  The actual value exists in
Gold (`trip_count`) but is not read at prediction time because the inference
path deliberately feeds a label-free Gold slice to the predictor.

`app/monitoring/actuals_join.py` already implements the backfill logic as a
CLI script, but it is not wired into any DAG.

## Scope

- Implement a daily `actuals_backfill_dag` in `orchestration/dags/`.
- The DAG calls `join_actuals(yesterday, yesterday, ...)` to fill the prior
  day's prediction history.
- Output written to `data/monitoring/joined/dt=<date>.parquet`.

## Why deferred

- Monitoring cadence (daily) is independent of inference cadence (15 min) —
  coupling them in the same DAG adds unnecessary Gold reads every tick.
- The joined output is also the intended input for the closed training-loop
  retraining trigger (mode-3, deferred).  Designing the schema before the
  training pipeline shape is known risks rework.
- Blocked on Monitoring ADR that defines the full monitoring surface
  (drift detection, accuracy metrics, retraining trigger thresholds).

## Related

- `app/monitoring/actuals_join.py` — implementation ready, needs DAG wrapper.
- `data/predictions/history/` — prediction logs written by `predict` task.
- Mode-3 (closed training loop) — deferred; will consume `joined/` output.
