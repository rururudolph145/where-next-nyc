# Design: Partitioned Prediction History + Watermark-Filtered /heatmap

**Date:** 2026-07-31
**Status:** Approved

---

## Problem

`Predictor.save_predictions` overwrites a single hot-slot Parquet file on every
pipeline tick. This means:

1. Historical prediction data is lost — monitoring, drift detection, and
   retrospective accuracy audits have no raw model output to work with.
2. `/heatmap` serves `hour.max()` from the loaded predictions DataFrame with no
   awareness of the pipeline's watermark position, so it can serve predictions
   from a future tick that hasn't "officially" published yet.

---

## Goals

- Accumulate prediction history in a day-partitioned store alongside the hot-slot.
- Gate `/heatmap`'s default hour selection to `<= watermark position`.
- Keep the hot-slot and API serving path unchanged (Phase 1).
- Lay the foundation for Phase 2: `/heatmap` reads directly from the partitioned
  history filtered by watermark (hot-slot dropped).

---

## Non-Goals

- Phase 2 implementation (reading from history instead of hot-slot).
- Any new API endpoints or config keys.
- Changes to the DAG task graph.

---

## Design

### Data Flow

```
[DAG tick]
    └─► Predictor.save_predictions(df)
            ├─► OVERWRITE  data/predictions/demand_predictions.parquet
            │               (hot-slot — unchanged, serves /heatmap)
            └─► WRITE      data/predictions/history/dt=<date>/part.parquet
                            (new — accumulates one file per day)

[GET /heatmap (no ?hour= param)]
    └─► _predictions_df (loaded from hot-slot at startup — unchanged)
            └─► filter: hour <= _current_sim_position() [watermark]
                    └─► take hour.max() within filtered set
```

The partition key `dt=<date>` is derived from the **prediction hour's date**, not
wall-clock date — e.g. predictions with `hour` values on `2021-03-01` go into
`dt=2021-03-01/`. One file per calendar day; ~25k rows (263 zones × 96 ticks/day).

### Touch Points

#### 1. `app/inference/predictor.py` — `save_predictions`

Add a day-partitioned history write after the existing hot-slot write.
The history write is wrapped in `try/except` — a failure logs a warning but
does not propagate, so the pipeline tick never fails due to history I/O.

```python
def save_predictions(self, predictions_df: pd.DataFrame) -> str:
    # existing hot-slot write (unchanged)
    os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    predictions_df.to_parquet(self.output_path, index=False)

    # new: day-partitioned history write
    try:
        for date, group in predictions_df.groupby(pd.Grouper(key="hour", freq="D")):
            partition_dir = os.path.join(
                os.path.dirname(self.output_path), "history", f"dt={date.date()}"
            )
            os.makedirs(partition_dir, exist_ok=True)
            group.to_parquet(os.path.join(partition_dir, "part.parquet"), index=False)
    except Exception:
        logger.warning("Failed to write prediction history — hot-slot unaffected")

    return self.output_path
```

The history path is derived from `self.output_path` — no new config key needed.

#### 2. `app/api/main.py` — `get_heatmap`

Replace the bare `hour.max()` fallback with a watermark-capped max.
`_current_sim_position()` already exists in `main.py` (used by `/health`).

```python
# before
latest = df["hour"].max()

# after
watermark = _current_sim_position()
filtered = df[df["hour"] <= pd.Timestamp(watermark)]
if filtered.empty:
    raise HTTPException(
        status_code=503,
        detail="No predictions available at or before the current watermark position.",
    )
latest = filtered["hour"].max()
```

The explicit `?hour=` path is unchanged — callers who pass a specific hour bypass
the watermark filter entirely.

#### 3. Config — no changes

The history path is derived from `inference.output_path` by appending `history/`.
No new keys in `config.yml`.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| History write fails | `try/except` logs warning; hot-slot write already succeeded; pipeline continues |
| Watermark behind all loaded predictions | 503: `"No predictions available at or before the current watermark position."` |
| Watermark file absent | `_current_sim_position()` falls back to `seed_start`; likely 503 on cold start — correct, no DAG has run yet |

**Nuance:** In steady-state live cadence the watermark *is* the pipeline's "now",
so predictions will always exist at or before it. The 503 paths are cold-start /
misconfiguration guards, not expected steady-state behaviour.

---

## Testing

### `Predictor.save_predictions` dual-write
- Happy path: assert hot-slot written; assert `history/dt=<date>/part.parquet` created with correct rows.
- Multi-day DataFrame: assert multiple partition dirs created.
- History write failure: mock history `to_parquet` to raise; assert hot-slot still written; assert no exception propagates from `save_predictions`.

### `get_heatmap` watermark filter
- Mock `_current_sim_position()` to a known datetime; assert `bucket_time <= watermark`.
- Watermark set before all prediction hours: assert 503 response.
- Explicit `?hour=` param: assert watermark filter is bypassed.

All tests are unit-testable with `tmp_path` fixtures and mocks — no Airflow or
external I/O required.

---

## Future: Phase 2

Once the partitioned history is proven stable (sufficient accumulation, no gaps),
`/heatmap` can be switched to read directly from `data/predictions/history/`
filtered by watermark. The hot-slot becomes redundant and is dropped. This change
requires no schema or API changes — only the load path in `lifespan` changes.
