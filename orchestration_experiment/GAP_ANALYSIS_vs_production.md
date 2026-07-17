# Gap Analysis — `orchestration_experiment/` vs `orchestration/`

> **Date:** 2026-07-17
> **Context:** After EXP-03b went GREEN (6-task chain: read_watermark → ingest →
> validate → feature → advance_watermark → write_watermark), a full comparison was
> done between the production DAG (`orchestration/dags/refresh_15min_dag.py`,
> Airflow 3.2.2, 11 tasks) and the current experiment stack. Gaps are scored
> **Urgent**, **Nice-to-have**, or **Unnecessary / over-engineering** from the
> perspective of the lean redesign goal.
>
> **Design intent:** EXP deliberately replaces the wall-clock idempotency model
> (`.done` sentinels) with a watermark-only model, and replaces the broken
> `_SpeedupClock` with data-watermark-as-simulation-position. Not every gap is
> a thing to port — several are the root cause of the over-engineering.

---

## DAG Tasks

| Gap | What it does in production | File / line | Score | Notes |
|-----|---------------------------|-------------|-------|-------|
| **`predict` task** | Runs `Predictor().run()` — loads model from MLflow registry, generates per-zone predictions, writes `demand_predictions.parquet` | `refresh_15min_dag.py:325–327` | **Urgent** | EXP-04. The whole point of the chain. |
| **`log_predict` task** | Reads predictions parquet, reshapes to `log_predictions()` format, appends to monitoring log | `refresh_15min_dag.py:329–332` | **Nice-to-have** | EXP-04. Needed for closed-loop monitoring but not for proving the pipeline runs. Skip gracefully if predictions empty. |
| **`check_no_new_data` task** | `AirflowSkipException` if watermark ≥ dataset max (`2021-03-31 23:59:57`) — prevents overrun | `refresh_15min_dag.py:305–308` | **Nice-to-have** | EXP-05. A single guard condition; could live in `advance_watermark` without a dedicated task. |
| **`decide_bucket` task** | Computes wall-clock `(day, bucket)` slot (0–95) to name the `.done` sentinel file | `refresh_15min_dag.py:297–300` | **Unnecessary** | Only exists to feed the `.done` sentinel model. EXP's watermark already prevents double-processing of the same data slice. |
| **`check_watermark` task** | Reads `{day}/{bucket:02d}.done`; skips the entire run if it exists | `refresh_15min_dag.py:309–312` | **Unnecessary** | The `.done` sentinel is a second idempotency layer on top of the data watermark. If the watermark advanced, the data was already processed. Two idempotency axes that can diverge is complexity, not safety. |
| **`write_done` task** | Writes `{day}/{bucket:02d}.done` after success (`trigger_rule="all_success"`) | `refresh_15min_dag.py:338–342` | **Unnecessary** | Only exists to support `check_watermark`. If we drop that, this drops too. |

> **Note on EXP-05:** The current EXP-05 PBI (`PBI_EXP05_idempotency_boundary.md`)
> plans to port `decide_bucket` / `check_watermark` / `write_done` from production.
> Given this analysis, **OQ2 in EXP-05 should be revisited**: the boundary stop
> (`check_no_new_data`) is worth adding, but the wall-clock sentinel trio is the
> over-engineering we are trying to escape. Consider EXP-05 scope = boundary only.

---

## Infrastructure

| Gap | What it does in production | File / line | Score | Notes |
|-----|---------------------------|-------------|-------|-------|
| **CeleryExecutor + celery-worker service** | Distributed task execution across multiple workers | `docker-compose.yaml:58, 174–195` | **Unnecessary** | One laptop, one DAG. LocalExecutor is the correct topology for this scope and removes Redis as a dependency. |
| **Redis service** | Message broker + result backend for CeleryExecutor | `docker-compose.yaml:108–120` | **Unnecessary** | Follows from CeleryExecutor; drops with it. |
| **Airflow triggerer service** | Handles async task deferral (e.g. `TimeDeltaSensor`, deferrable operators) | `docker-compose.yaml:197–210` | **Unnecessary** | No deferred sensors in the EXP chain. |
| **Celery Flower** | Worker monitoring UI (`--profile flower`) | `docker-compose.yaml:322–339` | **Unnecessary** | Follows from CeleryExecutor. |
| **DAG processor service** | Dedicated process for parsing DAGs and catching import errors | `docker-compose.yaml:159–172` | **Nice-to-have** | The scheduler handles this fine in LocalExecutor. Relevant at scale, not here. |
| **Debug CLI profile** | `--profile debug` drops a `airflow` CLI container for manual inspection | `docker-compose.yaml:304–316` | **Nice-to-have** | `docker exec` achieves the same thing. |

---

## Configuration & Environment

| Gap | What it does in production | File / line | Score | Notes |
|-----|---------------------------|-------------|-------|-------|
| **Dataset boundary constant** | `DATASET_MAX = datetime(2021, 3, 31, 23, 59, 57)` — stops the DAG at the end of the raw data | `refresh_15min_dag.py:54` | **Nice-to-have** | Worth adding as a constant in EXP-04/05 to prevent runaway watermark. One line. |
| **`.env` template** | `AIRFLOW_UID`, `FERNET_KEY`, credential override warning | `orchestration/README.md:20` (referenced but file absent from repo) | **Nice-to-have** | EXP has a hardcoded Fernet key in compose (acceptable for local-only use, not for sharing). Low priority. |
| **`airflow.cfg` override mount** | Custom parallelism / pool / timeout tuning via mounted file | `docker-compose.yaml:82`, `orchestration/config/` | **Unnecessary** | EXP compose env vars cover everything needed. Config file mount is production-operational overhead. |
| **`AIRFLOW__CORE__LOAD_EXAMPLES: 'true'`** | Loads built-in example DAGs on startup | `docker-compose.yaml:65` | **Unnecessary** | EXP already disables this correctly. |

---

## Monitoring & Observability

| Gap | What it does in production | File / line | Score | Notes |
|-----|---------------------------|-------------|-------|-------|
| **Scheduler health check** | HTTP readiness probe on port 8974 | `docker-compose.yaml:147–152` | **Nice-to-have** | Useful for knowing when the scheduler is actually ready before running `dags test`. One block in compose. |
| **Airflow API server service** | REST API (remapped to port 8090) — enables `af` CLI against local stack | `docker-compose.yaml:122–142` | **Nice-to-have** | Not needed for the `dags test` workflow but enables `af runs list` etc. |
| **Triggerer / worker health checks** | Readiness probes for async executor services | `docker-compose.yaml:197–210` | **Unnecessary** | Only relevant with CeleryExecutor. |

---

## DAG Semantics

| Gap | What it does in production | File / line | Score | Notes |
|-----|---------------------------|-------------|-------|-------|
| **`_clock_factory.py` / `_SpeedupClock`** | Process-global speedup clock for `DEMO_MODE=1`; `set_active_clock()` writes to a module-level variable | `orchestration/dags/_clock_factory.py` | **Unnecessary** | This is the thing that was broken (set in dag-processor, never seen by task workers → silently ran WallClock). EXP replaces it with watermark-as-simulation-position. **Do not port.** |
| **Wall-clock idempotency model** (`decide_bucket` + `check_watermark` + `write_done`) | Per `(day, bucket)` `.done` sentinel prevents re-triggering the same wall-clock slot | Multiple locations | **Unnecessary** | See DAG Tasks section above. The watermark-only model handles the same safety invariant without the extra state. |

---

## Planned Future DAGs (Not in Either Stack Yet)

| Gap | Notes |
|-----|-------|
| **`train_monthly_dag` (PBI 25c)** | Monthly training + MLflow registry gate. Roadmapped in `orchestration/README.md`. Not yet in EXP roadmap either. |
| **`monitor_hourly_dag` (PBI 25d)** | Drift detection + retrain trigger. Roadmapped in `orchestration/README.md`. Depends on PBIs 23/24. |

---

## Summary

| Score | Count | Items |
|-------|-------|-------|
| **Urgent** | 1 | `predict` task (EXP-04) |
| **Nice-to-have** | 7 | `log_predict`, `check_no_new_data`, dataset boundary constant, `.env` template, DAG processor service, scheduler health check, Airflow API server, debug CLI profile |
| **Unnecessary** | 9 | `decide_bucket`, `check_watermark`, `write_done`, CeleryExecutor + Redis, triggerer, Flower, `airflow.cfg` mount, load-examples toggle, `_clock_factory.py` |

The three tasks that account for most of the production DAG's perceived over-engineering are
`decide_bucket`, `check_watermark`, and `write_done` — a wall-clock idempotency layer that
is redundant given the watermark-only model. Removing them shrinks the final EXP DAG from
11 tasks to 8 (`decide_bucket` → `read_watermark` → `check_no_new_data` → `check_watermark`
→ ingest → validate → feature → predict → `log_predict` → `advance_watermark` → `write_done`
becomes `read_watermark` → `check_no_new_data` → ingest → validate → feature → predict →
`log_predict` → `advance_watermark` → `write_watermark`).

### Implication for EXP-05

The current `PBI_EXP05_idempotency_boundary.md` plans to port the full wall-clock sentinel
trio. Given this analysis, EXP-05 scope should be reconsidered: **add the boundary stop only**
(`check_no_new_data`), and close the PBI there. The wall-clock idempotency is the complexity
we are trying to eliminate, not preserve.
