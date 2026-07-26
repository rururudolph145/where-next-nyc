# PBI EXP-04 — Inference (Predict + Log) (Experimentation)

> **Type:** Experimentation PBI (not for release; lives under `orchestration_experiment/`)
> **Depends on:** EXP-03b (gold feature matrix must materialize; EXP-03a's
> validate gate is inherited transitively).
> **Parent:** Orchestrator redesign brainstorm (see `MEMORY.md → orchestrator-redesign-brainstorm`).

---

## 1. Why this PBI exists

EXP-03 produced gold. EXP-04 closes the mental model's "all the way to
inference": consume gold, predict per-zone demand for the next bucket, and log
the result for monitoring.

> At the target bucket `[W, W+slice)`, the model uses features built from
> history ending at `W` to predict demand per zone. Output →
> `demand_predictions.parquet`. Log for later scoring (PBI 22).

---

## 2. Goal

Insert `predict` and `log_predict` between `feature` and `advance_watermark`,
completing the end-to-end slice: one DAG run walks raw → bronze → silver →
gold → predictions in a single pass.

---

## 3. In scope

- Add `predict` + `log_predict`:
  `... → feature → predict → log_predict → advance_watermark → write_watermark`.
- `predict` — `Predictor().run()` (gold → predictions parquet).
- `log_predict` — read predictions parquet, reshape into the response dict
  shape `log_predictions` expects, call `log_predictions(...)`. Skip
  gracefully (warn, return) if predictions file is missing or empty.
- Outputs under the isolated experiment data root.

## 4. Out of scope (deferred)

- `.done` sentinels + `check_no_new_data` boundary — EXP-05.
- Serving (`/heatmap`, `/predict`) — that's `app.api`, already shipped; not
  part of the DAG experiment.
- Retraining — separate future DAG (watermark-delta trigger, per brainstorm).
- Any change to `app/`, `orchestration/`, `implementation_checklist/`.

---

## 5. Deliverables

1. Updated DAG with `predict` + `log_predict` inserted — full linear chain.
2. `demand_predictions.parquet` under the experiment data root after a run.
3. Monitoring log entries written under the experiment data root.
4. Note recording: predictions have one row per `(zone, target_bucket)`;
   target bucket == `[W, W+slice)`; log entries present.

---

## 6. DAG shape

```
read_watermark → ingest → validate → feature → predict → log_predict
  → advance_watermark → write_watermark
```

This is the full end-to-end experiment chain (minus idempotency, which is
EXP-05). It mirrors the production `refresh_15min_dag` task order, sans the
`decide_bucket` / `check_watermark` / `check_no_new_data` gates.

---

## 7. Acceptance (experimental)

- After a run: `demand_predictions.parquet` exists, one row per
  `(zone, target_bucket)`, target bucket aligned to the current window.
- `log_predict` writes a monitoring entry; empty/missing predictions →
  graceful skip (warn, no crash).
- A failure anywhere upstream of `advance_watermark` leaves the watermark
  unchanged (end-to-end crash safety).
- No imports from `orchestration/`.

---

## 8. Open questions (log answers here as we go)

- **OQ1** — Which model artifact does `Predictor` load in the experiment? A
  copy under the experiment root, or the shared `models/`? (Leaning: shared
  read-only `models/` — we're not retraining here.)
- **OQ2** — Does the prediction horizon match the target bucket exactly
  (`[W, W+slice)`), or is there an offset? Confirm against `Predictor` and note
  it, since it defines what `/heatmap` would show.
