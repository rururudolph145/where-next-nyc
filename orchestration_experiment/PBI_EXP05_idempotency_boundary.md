# PBI EXP-05 — Idempotency Sentinels + Boundary Sensor (Experimentation)

> **Type:** Experimentation PBI (not for release; lives under `orchestration_experiment/`)
> **Depends on:** EXP-04 (full end-to-end chain must run).
> **Parent:** Orchestrator redesign brainstorm (see `MEMORY.md → orchestrator-redesign-brainstorm`).

---

## 1. Why this PBI exists

EXP-04 gave us a full raw → predictions chain, but it will happily reprocess
the same wall-clock slot and will run off the end of the dataset. EXP-05 adds
the two guards the production DAG has:

1. **Wall-clock idempotency** via a per-`(day, bucket)` `.done` sentinel
   (`decide_bucket` / `check_watermark` / `write_done`).
2. **Data boundary** via `check_no_new_data` — stop advancing once the
   watermark reaches the end of the raw dataset.

This is the "make it safe to re-trigger and safe to exhaust" PBI.

---

## 2. Goal

Wrap the EXP-04 chain with the idempotency and boundary gates so that:

- Re-triggering the same wall-clock 15-min slot is a no-op.
- Reaching the dataset's max timestamp skips cleanly (no error, no runaway
  watermark).

---

## 3. In scope

- `decide_bucket` — compute `(day, bucket)` from the real wall clock
  (`bucket = hour*4 + minute//15`, 0..95). Drives the `.done` sentinel ONLY.
- `check_no_new_data` — `AirflowSkipException` if `window_start >= dataset_max`.
- `check_watermark` — `AirflowSkipException` if `(day, bucket)` `.done` exists.
- `write_done` — mark `(day, bucket)` complete after success
  (`trigger_rule="all_success"`).
- Sentinels under the isolated experiment data root
  (e.g. `orchestration_experiment/data/watermarks/`).

## 4. Out of scope (deferred / not planned)

- Config loading refinements beyond what EXP-02/03/04 established.
- Retraining DAG — separate future work.
- Promotion of the experiment DAG back into `orchestration/dags/` — that is a
  later, non-experimentation decision once EXP-01..05 are proven.
- Any change to `app/`, `orchestration/`, `implementation_checklist/`.

---

## 5. Deliverables

1. Final experiment DAG matching production task order (11 tasks).
2. `.done` sentinels written per `(day, bucket)` under the experiment root.
3. Note recording: re-trigger of a done bucket skips; watermark stops at
   dataset max; sentinels accumulate one per processed wall-clock slot.

---

## 6. DAG shape (final, mirrors production)

```
decide_bucket → read_watermark → check_no_new_data → check_watermark
  → ingest → validate → feature → predict → log_predict
  → advance_watermark → write_done
```

Note the two idempotency axes remain independent (confirmed with user
2026-07-14):
- **Wall-clock position** (`decide_bucket`) → drives `.done` sentinel.
- **Data-time position** (`read_watermark` watermark) → drives the ingest
  window. They must NOT be conflated (the old `_SpeedupClock` DAG's mistake).

---

## 7. Acceptance (experimental)

- Trigger the same wall-clock slot twice → second run skips at
  `check_watermark`; no duplicate processing.
- Advance watermark to `dataset_max` → next run skips at `check_no_new_data`;
  watermark does not overrun.
- A successful run writes exactly one `.done` sentinel for its `(day, bucket)`.
- Full chain still produces predictions on a fresh, not-yet-done slot.
- No imports from `orchestration/`.

---

## 8. Open questions (log answers here as we go)

- **OQ1** — `dataset_max` source: hardcode (as production does with
  `2021-03-31 23:59:57`) or derive from the raw parquet at runtime? (Leaning:
  hardcode in experiment config for determinism; note the value.)
- **OQ2** — Once EXP-05 passes, do we promote this DAG into
  `orchestration/dags/` (replacing the current `refresh_15min_dag.py`), or
  keep iterating? Decision deferred to a post-experiment review.
