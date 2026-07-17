# Implementation Plan — PBI EXP-03b (Feature, Silver → Gold)

> **Source PBI:** `orchestration_experiment/PBI_EXP03b_feature_silver_gold.md`
> **Depends on:** EXP-03a (validate gate — GREEN, verified 2026-07-17).
> **Stack:** Airflow **2.10.3** LocalExecutor, custom Dockerfile.
> **Created:** 2026-07-17

---

## Overview

Insert one `feature` task between `validate` and `advance_watermark`. The task
runs `DataCleaner(config).run()` (bronze → silver) then
`FeatureEngineer(config).run()` (silver → gold) behind a single label, using an
injected override config that redirects `data.silver_path`, `data.gold_path`,
`feature.zone_shapefile_path`, and `model.store_path` to the experiment data
root. Both `app` classes and every causal transform they call are already
covered by existing unit tests — EXP-03b writes **no new domain logic**, only
the config-override seam (`build_feature_config`), a thin `run_feature` wrapper,
and the DAG plumbing. Built RED → GREEN like EXP-02/03a.

---

## Decisions locked (with the user, 2026-07-17)

1. **OQ1 — inject config dict.** Same seam as EXP-02/03a. `DataCleaner(config)`
   takes config positionally (required); `FeatureEngineer(config=None)` takes it
   as a keyword — one override dict satisfies both. No env vars.
2. **OQ2 — manual spot-check in the note, no new leakage test.** The causal
   property lives in `FeatureEngineer.add_lag_features` /
   `add_shifted_target`, already covered by
   `tests/test_feature.py::TestShiftedTarget` and `TestNeighborLagSmoothing`.
   The experiment DAG *calls* those methods, doesn't reimplement them —
   re-testing through a data-heavy path adds cost, not confidence. We document
   one `(zone, bucket)` spot-check in the PBI verification note.
3. **OQ3 — redirect `model.store_path`** to `data_exp/models/`. `DataCleaner.run`
   persists `iqr_bounds.json` there; the file is small and a future training DAG
   will want it. The only hard rule: it must NOT be the production `models/`.
4. **OQ4 — override `inference.simulation.start_hour` to `window_end`** in
   `build_feature_config`. The experiment window (`[2021-01-18, …)` advancing
   15 min/run) is entirely before the production cutoff `2021-03-01`, so
   `DataCleaner.run`'s train/inference split would otherwise classify every row
   as "train" and hand the IQR-application step an empty inference frame.
   Pinning the cutoff to `window_end` makes the split well-defined every run and
   provably cannot hit the empty-inference edge. Costs one line.

---

## Key findings from investigation

- **`app/ingestion/clean.py::DataCleaner`** —
  `__init__(config)` (positional, required) reads `env`, `data.bronze_path`,
  `data.silver_path`, `clean.null_strategy`, `clean.null_subset` (optional).
  `run() -> pd.DataFrame` reads bronze from `self.bronze_path`, validates schema,
  splits train/inference at `config["inference"]["simulation"]["start_hour"]`,
  fits IQR bounds on train only, applies to inference, writes silver to
  `self.silver_path`, and persists `iqr_bounds.json` under
  `config["model"]["store_path"]`.
- **`app/ingestion/feature.py::FeatureEngineer`** —
  `__init__(config=None)` (loads via `load_config()` if None) reads
  `feature.bucket`, `feature.lags`, `feature.horizon_buckets`,
  `feature.horizon_sum_buckets`, `feature.n_neighbors`,
  `feature.zone_shapefile_path`, `data.silver_path`, `data.gold_path`.
  `run(data=None) -> pd.DataFrame` reads silver when `data is None`, chains
  hourly-demand → time → cyclical → lag → (shifted/summed target) →
  `add_zone_coordinates` → (neighbor smoothing), writes
  `<gold_path>/hourly_demand_featured.parquet`.
- **Gold grain and columns (the real names — the PBI's `(zone, target_bucket)`
  maps to these):**
  - grain: one row per **`(pulocationid, hour)`** bucket.
  - demand: `pulocationid`, `hour`, `trip_count`.
  - cyclical: `hour_of_day_sin/cos`, `day_of_week_sin/cos`, `month_sin/cos`
    (plus raw `hour_of_day`, `day_of_week`, `month`).
  - lags (from `feature.lags=[1,2,4,8,12,24,96,672]`): `trip_count_lag_1 … _672`.
  - summed-horizon target (since `horizon_sum_buckets=2`): **`target_sum_t1_t2`**.
  - zone coords: `pickup_latitude`, `pickup_longitude` (WGS84 centroids;
    orphan zones 57/264/265 dropped).
  - neighbor smoothing (since `n_neighbors=5`):
    `trip_count_lag_1_neighbor_mean`.
- **`add_zone_coordinates`** calls `build_centroid_lookup(self.zone_shapefile_path)`
  → reads the shapefile via geopandas, projects, returns
  `{LocationID: (lat, lon)}`. Missing-coordinate rows are dropped. This is why
  the shapefile mount is required.
- **`config.yml` current values** (production, must NOT be written to):
  `data.silver_path=data/silver/yellow_tripdata_cleaned.parquet`,
  `data.gold_path=data/gold/yellow_tripdata`, `model.store_path=models/`,
  `feature.zone_shapefile_path=data/reference/taxi_zones.shp`,
  `feature.horizon_sum_buckets=2`, `feature.bucket=15min`,
  `inference.simulation.start_hour=2021-03-01 00:00:00`.
- **EXP-03a is the canonical precedent** for the file layout: pure helper
  (`exp03a_validate.py`) + thin DAG wrapper (`exp03a_validate_dag.py`) with the
  dual-context import, structural test guarded by `pytest.importorskip`, and
  helper tests that mock at the module boundary.
- **`geopandas` / `scikit-learn` are NOT in the baked EXP image** (EXP-02/03a
  never needed them). EXP-03b adds both.

---

## Affected Files

- `orchestration_experiment/dags/exp03b_feature.py` — **new** pure/Airflow-free
  helper: `build_feature_config`, `run_feature`.
- `orchestration_experiment/dags/exp03b_feature_dag.py` — **new** DAG wrapper.
  6-task linear chain; dual-context import; `_feature` pulls `bronze_path` from
  `ingest`'s XCom (silver/gold paths come from the override config, not XCom).
- `orchestration_experiment/requirements-exp.txt` — **update**: add
  `geopandas>=1.0` and `scikit-learn>=1.5` (match `pyproject.toml` pins). Update
  the header comment to note EXP-03b adds them.
- `orchestration_experiment/docker-compose-LocalExecutor.yml` — **update**: add
  a read-only mount `../data/reference:/opt/airflow/data/reference:ro` so
  `taxi_zones.shp` is readable by `add_zone_coordinates`. (The writable
  `./data:/opt/airflow/data_exp` mount already covers silver/gold/models.)
- `orchestration_experiment/Dockerfile` — no code change; rebuild picks up deps.
- `tests/test_exp03b_feature.py` — **new** RED-first unit tests (source below).
- `tests/test_exp03b_dag_structure.py` — **new** 6-task structural test with
  `pytest.importorskip("airflow")`; mirrors `test_exp03a_dag_structure.py`.
- `orchestration_experiment/PBI_EXP03b_feature_silver_gold.md` — **update**:
  append verification note (OQ answers, smoke results, causality spot-check)
  after GREEN.

---

## The failing test (RED source — write this file first, verbatim)

Save as `tests/test_exp03b_feature.py`. Every test starts as a
`ModuleNotFoundError` until the helper lands. Tests mock `DataCleaner` and
`FeatureEngineer` at the module boundary — we assert the **wiring** (correct
config injected, correct call order, silver-then-gold), NOT the transforms
themselves (those are covered in `tests/test_feature.py`).

```python
"""Tests for orchestration_experiment/dags/exp03b_feature.py.

PBI EXP-03b (experimentation) — the pure helpers the feature DAG delegates to.
Inserts a ``feature`` task between ``validate`` and ``advance_watermark``:
``DataCleaner(config).run()`` (bronze → silver) then
``FeatureEngineer(config).run()`` (silver → gold), behind one task label because
they share the same override config and always run together.

The logic lives in a PURE, Airflow-free module (the experiment container has
Airflow; the host test env does NOT). Two seams keep it unit-testable without
reading a real shapefile or a 340MB bronze:

  * ``build_feature_config`` — deep-overrides ``data.silver_path``,
    ``data.gold_path``, ``feature.zone_shapefile_path``, ``model.store_path``
    (redirect to the experiment root), and ``inference.simulation.start_hour``
    (pin to ``window_end`` so DataCleaner's train/inference split is
    well-defined for the pre-cutoff experiment window — OQ4). Nothing else
    changes.
  * ``run_feature`` — constructs ``DataCleaner(config)``, calls ``.run()``, then
    ``FeatureEngineer(config).run()``, and returns the bronze path unchanged
    (stable XCom flow). We assert order and config injection by mocking both
    classes at the module boundary.

AC coverage:
  - build_feature_config overrides the four experiment paths
  - build_feature_config pins inference.simulation.start_hour to window_end (OQ4)
  - build_feature_config does not mutate the base config (deep copy)
  - build_feature_config preserves unrelated keys
  - run_feature constructs DataCleaner with the injected config and calls run()
  - run_feature constructs FeatureEngineer with the injected config and calls run()
  - run_feature runs the cleaner BEFORE the feature engineer (silver→gold order)
  - run_feature returns the bronze path (stable XCom contract)
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchestration_experiment.dags.exp03b_feature import (
    build_feature_config,
    run_feature,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

BRONZE_PATH = "/opt/airflow/data_exp/bronze/yellow_tripdata/bronze.parquet"
SILVER_ROOT = "/opt/airflow/data_exp/silver/yellow_tripdata_cleaned.parquet"
GOLD_ROOT = "/opt/airflow/data_exp/gold/yellow_tripdata"
SHAPEFILE = "/opt/airflow/data/reference/taxi_zones.shp"
MODEL_ROOT = "/opt/airflow/data_exp/models"
WINDOW_END = datetime(2021, 1, 18, 0, 15, 0)


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Minimal config mirroring the shape both classes read."""
    return {
        "env": "experiment",
        "ingest": {"slice_duration": "15min"},
        "clean": {"null_strategy": "drop"},
        "feature": {
            "bucket": "15min",
            "lags": [1, 2],
            "horizon_sum_buckets": 2,
            "n_neighbors": 5,
            "zone_shapefile_path": "data/reference/taxi_zones.shp",
        },
        "data": {
            "bronze_path": "data/bronze/yellow_tripdata/bronze.parquet",
            "silver_path": "data/silver/yellow_tripdata_cleaned.parquet",
            "gold_path": "data/gold/yellow_tripdata",
        },
        "model": {"store_path": "models/"},
        "inference": {"simulation": {"start_hour": "2021-03-01 00:00:00"}},
    }


@pytest.fixture
def override_config(base_config: dict[str, Any]) -> dict[str, Any]:
    """A config already redirected to the experiment roots."""
    return build_feature_config(
        base_config,
        silver_root=SILVER_ROOT,
        gold_root=GOLD_ROOT,
        shapefile_path=SHAPEFILE,
        model_root=MODEL_ROOT,
        window_end=WINDOW_END,
    )


# ---------------------------------------------------------------------------
# TestBuildFeatureConfig
# ---------------------------------------------------------------------------


class TestBuildFeatureConfig:
    """build_feature_config injects the isolated paths + OQ4 cutoff (the seam)."""

    def test_overrides_the_four_paths(self, base_config: dict[str, Any]) -> None:
        """silver/gold/shapefile/model paths point at the experiment roots."""
        # Act
        result = build_feature_config(
            base_config,
            silver_root=SILVER_ROOT,
            gold_root=GOLD_ROOT,
            shapefile_path=SHAPEFILE,
            model_root=MODEL_ROOT,
            window_end=WINDOW_END,
        )

        # Assert
        assert result["data"]["silver_path"] == SILVER_ROOT
        assert result["data"]["gold_path"] == GOLD_ROOT
        assert result["feature"]["zone_shapefile_path"] == SHAPEFILE
        assert result["model"]["store_path"] == MODEL_ROOT

    def test_pins_start_hour_to_window_end(
        self, base_config: dict[str, Any]
    ) -> None:
        """OQ4: the train/inference cutoff is pinned to window_end."""
        # Act
        result = build_feature_config(
            base_config,
            silver_root=SILVER_ROOT,
            gold_root=GOLD_ROOT,
            shapefile_path=SHAPEFILE,
            model_root=MODEL_ROOT,
            window_end=WINDOW_END,
        )

        # Assert
        assert result["inference"]["simulation"]["start_hour"] == (
            WINDOW_END.isoformat()
        )

    def test_does_not_mutate_base(self, base_config: dict[str, Any]) -> None:
        """The base config is deep-copied, never mutated in place."""
        # Arrange
        snapshot = copy.deepcopy(base_config)

        # Act
        build_feature_config(
            base_config,
            silver_root=SILVER_ROOT,
            gold_root=GOLD_ROOT,
            shapefile_path=SHAPEFILE,
            model_root=MODEL_ROOT,
            window_end=WINDOW_END,
        )

        # Assert
        assert base_config == snapshot

    def test_preserves_unrelated_keys(self, base_config: dict[str, Any]) -> None:
        """Only the redirected keys change; everything else passes through."""
        # Act
        result = build_feature_config(
            base_config,
            silver_root=SILVER_ROOT,
            gold_root=GOLD_ROOT,
            shapefile_path=SHAPEFILE,
            model_root=MODEL_ROOT,
            window_end=WINDOW_END,
        )

        # Assert
        assert result["data"]["bronze_path"] == base_config["data"]["bronze_path"]
        assert result["clean"] == base_config["clean"]
        assert result["feature"]["lags"] == base_config["feature"]["lags"]
        assert (
            result["feature"]["horizon_sum_buckets"]
            == base_config["feature"]["horizon_sum_buckets"]
        )


# ---------------------------------------------------------------------------
# TestRunFeature
# ---------------------------------------------------------------------------


class TestRunFeature:
    """run_feature wires DataCleaner → FeatureEngineer with the override config."""

    def test_constructs_cleaner_with_injected_config(
        self, override_config: dict[str, Any]
    ) -> None:
        """DataCleaner is built with the override config and run() is called."""
        # Arrange
        with (
            patch(
                "orchestration_experiment.dags.exp03b_feature.DataCleaner"
            ) as mock_cleaner,
            patch(
                "orchestration_experiment.dags.exp03b_feature.FeatureEngineer"
            ),
        ):
            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert
        mock_cleaner.assert_called_once_with(override_config)
        mock_cleaner.return_value.run.assert_called_once_with()

    def test_constructs_engineer_with_injected_config(
        self, override_config: dict[str, Any]
    ) -> None:
        """FeatureEngineer is built with the override config and run() is called."""
        # Arrange
        with (
            patch("orchestration_experiment.dags.exp03b_feature.DataCleaner"),
            patch(
                "orchestration_experiment.dags.exp03b_feature.FeatureEngineer"
            ) as mock_engineer,
        ):
            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert
        mock_engineer.assert_called_once_with(config=override_config)
        mock_engineer.return_value.run.assert_called_once_with()

    def test_cleaner_runs_before_engineer(
        self, override_config: dict[str, Any]
    ) -> None:
        """Silver must exist before gold: cleaner.run() precedes engineer.run()."""
        # Arrange
        manager = MagicMock()
        with (
            patch(
                "orchestration_experiment.dags.exp03b_feature.DataCleaner",
                manager.DataCleaner,
            ),
            patch(
                "orchestration_experiment.dags.exp03b_feature.FeatureEngineer",
                manager.FeatureEngineer,
            ),
        ):
            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert — the cleaner's run() call is recorded before the engineer's
        call_names = [c[0] for c in manager.mock_calls]
        cleaner_run = call_names.index("DataCleaner().run")
        engineer_run = call_names.index("FeatureEngineer().run")
        assert cleaner_run < engineer_run

    def test_returns_bronze_path(self, override_config: dict[str, Any]) -> None:
        """A stable XCom contract: run_feature hands the bronze path forward."""
        # Arrange
        with (
            patch("orchestration_experiment.dags.exp03b_feature.DataCleaner"),
            patch("orchestration_experiment.dags.exp03b_feature.FeatureEngineer"),
        ):
            # Act
            result = run_feature(BRONZE_PATH, config=override_config)

        # Assert
        assert result == BRONZE_PATH
```

### Expected RED output

```
$ uv run pytest tests/test_exp03b_feature.py
...
ModuleNotFoundError: No module named 'orchestration_experiment.dags.exp03b_feature'
```

If ANY test passes, or the ImportError names something other than
`exp03b_feature`, stop — the RED is not proving what we think. Fix the test file
before writing production code.

---

## Implementation Steps

**Phase 1 — RED**

1. [ ] **Write `tests/test_exp03b_feature.py`** verbatim from the block above.
2. [ ] **Verify RED** — `uv run pytest tests/test_exp03b_feature.py`. Confirm the
   failure is `ModuleNotFoundError: No module named
   'orchestration_experiment.dags.exp03b_feature'`. This is the "watched it fail
   for the right reason" gate.

**Phase 2 — GREEN (helper)**

3. [ ] **Implement `orchestration_experiment/dags/exp03b_feature.py`** — minimum
   code to flip every RED test to GREEN. Shape:

   ```python
   """Pure feature helpers for the EXP-03b silver→gold DAG.

   Airflow-free by design (the host test env has no Airflow). The DAG file
   (``exp03b_feature_dag.py``) is a thin wrapper that pulls ``bronze_path`` from
   XCom, builds the override config, and calls ``run_feature``.

   Two seams keep this unit-testable:

     * :func:`build_feature_config` redirects silver/gold/shapefile/model paths
       to the experiment root and pins the train/inference cutoff to
       ``window_end`` (OQ4 — the pre-cutoff experiment window would otherwise
       leave DataCleaner's inference split empty).
     * :func:`run_feature` runs ``DataCleaner(config).run()`` (bronze→silver)
       then ``FeatureEngineer(config).run()`` (silver→gold), returning the
       bronze path unchanged for a stable XCom flow.
   """

   from __future__ import annotations

   import copy
   from datetime import datetime
   from typing import Any

   from app.ingestion.clean import DataCleaner
   from app.ingestion.feature import FeatureEngineer


   def build_feature_config(
       base: dict[str, Any],
       *,
       silver_root: str,
       gold_root: str,
       shapefile_path: str,
       model_root: str,
       window_end: datetime,
   ) -> dict[str, Any]:
       """Return a copy of *base* redirected to the experiment roots.

       Overrides four output paths plus the train/inference cutoff (OQ4). The
       base is deep-copied, never mutated.
       """
       config = copy.deepcopy(base)
       config["data"]["silver_path"] = silver_root
       config["data"]["gold_path"] = gold_root
       config["feature"]["zone_shapefile_path"] = shapefile_path
       config["model"]["store_path"] = model_root
       config["inference"]["simulation"]["start_hour"] = window_end.isoformat()
       return config


   def run_feature(bronze_path: str, *, config: dict[str, Any]) -> str:
       """Clean bronze→silver, then engineer silver→gold. Return bronze_path.

       DataCleaner takes config positionally; FeatureEngineer takes it as a
       keyword. The cleaner MUST run first — the engineer reads the silver it
       writes. bronze_path is returned unchanged so the DAG's XCom flow
       (ingest → validate → feature → advance) stays predictable.
       """
       DataCleaner(config).run()
       FeatureEngineer(config=config).run()
       return bronze_path
   ```

4. [ ] **Verify GREEN** — `uv run pytest tests/test_exp03b_feature.py`. All pass,
   output pristine. Then `uv run pytest` full suite: baseline unchanged apart
   from the new tests.

**Phase 3 — RED then GREEN for DAG structure**

5. [ ] **Write `tests/test_exp03b_dag_structure.py`** — copy
   `tests/test_exp03a_dag_structure.py` verbatim; change:
   - module docstring to reference EXP-03b and step 7 below;
   - the import: `from orchestration_experiment.dags.exp03b_feature_dag import dag`;
   - `assert dag.dag_id == "exp03b_feature_dag"`;
   - rename the count test to `test_exactly_six_tasks` — expected set:
     `{"read_watermark", "ingest", "validate", "feature", "advance_watermark",
     "write_watermark"}`;
   - `test_linear_dependency_chain` — `feature.upstream == {"validate"}`,
     `advance.upstream == {"feature"}`;
   - keep `test_schedule_is_15_min`, `test_no_catchup_single_active_run`,
     `test_advance_and_write_require_all_success`.

6. [ ] **Verify RED** — `uv run pytest tests/test_exp03b_dag_structure.py`. On
   the host Airflow is absent, so `pytest.importorskip("airflow")` skips the
   file — that IS the host-side RED. Structural correctness is authoritatively
   verified in-container in step 12.

7. [ ] **Implement `orchestration_experiment/dags/exp03b_feature_dag.py`** — clone
   `exp03a_validate_dag.py`; then:
   - Add dual-context import for the new helper:
     ```python
     try:
         from orchestration_experiment.dags.exp03b_feature import (
             build_feature_config, run_feature,
         )
     except ModuleNotFoundError:
         from exp03b_feature import (  # type: ignore[no-redef,import-not-found]
             build_feature_config, run_feature,
         )
     ```
   - Add constants:
     ```python
     GOLD_ROOT = "/opt/airflow/data_exp/gold/yellow_tripdata"
     SHAPEFILE_PATH = "/opt/airflow/data/reference/taxi_zones.shp"
     MODEL_ROOT = "/opt/airflow/data_exp/models"
     ```
     (`SILVER_ROOT` already exists from the EXP-03a clone.)
   - Add `_feature` callable — it needs `window_end` for the OQ4 cutoff, pulled
     from `read_watermark`'s XCom (the same window `_advance` reads):
     ```python
     def _feature(**context: object) -> None:
         ti = context["ti"]  # type: ignore[index]
         bronze_path: str = ti.xcom_pull(task_ids="ingest")  # type: ignore[attr-defined]
         window: dict[str, str] = ti.xcom_pull(task_ids="read_watermark")  # type: ignore[attr-defined]
         window_end = datetime.fromisoformat(window["window_end"])
         config = build_feature_config(
             load_config(),
             silver_root=SILVER_ROOT,
             gold_root=GOLD_ROOT,
             shapefile_path=SHAPEFILE_PATH,
             model_root=MODEL_ROOT,
             window_end=window_end,
         )
         run_feature(bronze_path, config=config)
         logger.info("feature: silver+gold materialized for %s", bronze_path)
     ```
   - Insert `feature = PythonOperator(task_id="feature",
     python_callable=_feature)` and rewire:
     `read_watermark >> ingest >> validate >> feature >> advance_watermark >>
     write_watermark`. `advance`/`write` keep `trigger_rule="all_success"`.
   - Change `dag_id` to `exp03b_feature_dag`; update `tags` to include
     `"feature"`, `"gold"`.

8. [ ] **Verify GREEN** — `uv run pytest tests/test_exp03b_dag_structure.py`. On
   host: still skipped (Airflow absent). In-container: step 12 exercises it.

**Phase 4 — build, boot, smoke**

9. [ ] **Deps + rebuild** — add `geopandas>=1.0` and `scikit-learn>=1.5` to
   `orchestration_experiment/requirements-exp.txt` (match `pyproject.toml`
   pins — verify the exact minimums there first). Update the header comment
   ("EXP-03b introduces geopandas + scikit-learn for zone centroids and
   neighbour smoothing"). Then `docker compose -f
   orchestration_experiment/docker-compose-LocalExecutor.yml build`. Note the
   image-size delta (geopandas + its GDAL/GEOS stack is the biggest fatten
   since EXP-02).

10. [ ] **Add the read-only reference mount** — in
    `docker-compose-LocalExecutor.yml`, under the volumes of every service that
    runs tasks (scheduler; and webserver/worker if present), add:
    `- ../data/reference:/opt/airflow/data/reference:ro`. Confirm the host path
    `data/reference/taxi_zones.shp` exists (it does — verified 2026-07-17).

11. [ ] **Permissions reset + boot (AIRFLOW QUIRK)** — same UID-50000 gotcha as
    EXP-02/03a: `docker compose ... exec -u 0 airflow-scheduler chown -R
    50000:0 /opt/airflow/data_exp`. Reseed `data/watermark.json` to
    `2021-01-18T00:00:00` if it advanced during EXP-03a smoke. `docker compose
    ... up -d`. FIRST command after boot:
    `airflow dags list-import-errors` — fix anything it returns before touching
    the UI. (The reference mount is read-only, so no chown needed there.)

12. [ ] **Smoke test (happy path)** — `airflow dags test exp03b_feature_dag
    2021-01-18`. Verify:
    - watermark advances `2021-01-18T00:00 → 00:15`;
    - `data_exp/silver/yellow_tripdata_cleaned.parquet` exists;
    - `data_exp/gold/yellow_tripdata/hourly_demand_featured.parquet` exists;
    - `data_exp/models/iqr_bounds.json` exists (OQ3 — NOT under production
      `models/`);
    - the OQ4 all-train path did NOT crash (task state = success).
    Then read the gold parquet in-container and record for the note:
    - one row per `(pulocationid, hour)` (no duplicate pairs);
    - required columns present: `trip_count_lag_1`, `trip_count_lag_2`,
      the cyclical `*_sin/*_cos` set, `pickup_latitude`, `pickup_longitude`,
      `target_sum_t1_t2`, `trip_count_lag_1_neighbor_mean`;
    - `max(hour) < window_end` and `min(hour) >= seed_start (2021-01-18)`;
    - **causality spot-check (OQ2):** pick one `(pulocationid, hour=t)`; assert
      `trip_count_lag_1` there equals `trip_count` at `(same zone, hour=t-1
      bucket)`.

13. [ ] **Smoke test (crash-safety, re-verifying the chain via EXP-03a's gate)** —
    back up bronze (`cp bronze.parquet bronze.parquet.bak`), corrupt
    `PULocationID` to `999` on all rows in-container, back up the current gold
    (`cp -r gold gold.bak`), then rerun `airflow dags test exp03b_feature_dag
    2021-01-19`. Verify:
    - `validate` state = `failed`;
    - `feature`, `advance_watermark`, `write_watermark` = `upstream_failed`
      (never ran);
    - gold parquet unchanged from step 12 (diff against `gold.bak`);
    - `data/watermark.json` unchanged from step 12's position;
    - restore: `mv bronze.parquet.bak bronze.parquet`, `rm -rf gold.bak`.

**Phase 5 — REFACTOR + gates**

14. [ ] **REFACTOR** — read `exp03b_feature.py` with fresh eyes (~25 lines).
    Likely nothing to do. If a docstring omits the silver-before-gold ordering
    contract, add one line. Rerun `uv run pytest tests/test_exp03b_feature.py` —
    must stay green.

15. [ ] **Boundary AC** — `grep -R "from orchestration"
    orchestration_experiment/dags/exp03b_* tests/test_exp03b_*` returns empty
    (the `try/except` dual-import references `orchestration_experiment.dags.…`
    for the HOST path — that's the test-import package path, allowed; the app
    import ban is about `app/` importing `orchestration/`, which we don't).
    Confirm
    `tests/test_pbi25a_orchestrator_skeleton.py::test_app_does_not_import_from_orchestration`
    still passes.

16. [ ] **Full quality gates** — `uv run ruff check .`, `uv run ruff format
    --check .`, `uv run mypy app tests`, `uv run pytest`. Baseline before this
    PBI: whatever EXP-03a left green. New count = baseline + 8 (helper tests) +
    6 (structural, skipped on host). Adjust to the actual counts.

17. [ ] **Fill PBI verification note** — append to
    `PBI_EXP03b_feature_silver_gold.md`:
    - **OQ1** → inject config dict (both classes accept it).
    - **OQ2** → causality spot-check documented inline (the one `(zone, bucket)`
      from step 12), trusting `tests/test_feature.py` for full coverage.
    - **OQ3** → `iqr_bounds.json` landed under `data_exp/models/`, not
      production.
    - **OQ4** → cutoff pinned to `window_end`; all-train path did not crash.
    - Both smoke results (silver/gold/model paths, row counts, column list,
      `max(hour)`/`min(hour)`, crash-safety states).
    - Do NOT commit. Per the `MEMORY.md` user preference, stop after GREEN for
      review.

---

## Risks & Considerations

- **RED-step OQ lock-in** — the failing test's `build_feature_config` signature
  (five keyword overrides incl. `window_end`) commits us to OQ1 + OQ4 before
  implementation. Reversing means rewriting the tests. Accepted — both are
  locked with the user above.
- **geopandas image fatten** — geopandas pulls GDAL/GEOS/pyproj; this is the
  largest image growth since EXP-02. One-time rebuild in step 9, cached
  thereafter. If the build is slow or the wheel resolution fights, pin exact
  versions from `uv pip compile` output rather than `>=`.
- **Shapefile mount is a hard dependency** — `add_zone_coordinates` crashes if
  `taxi_zones.shp` is unreadable. The mount is `:ro` and points at the same
  sidecar files (`.shp/.shx/.dbf/.prj`) geopandas needs — all four must be
  present in `data/reference/` (verified). A missing sidecar surfaces as an
  opaque fiona error; check all four exist if `feature` fails at coord-lookup.
- **OQ4 empty-inference edge** — pinning the cutoff to `window_end` means the
  inference split is `hour >= window_end`, which is empty by construction (all
  ingested buckets are `< window_end`). This is the SAME all-train situation,
  just made explicit and deterministic rather than dependent on a far-future
  literal. If `DataCleaner.run` still chokes on a zero-row inference frame,
  fall back to a cutoff *inside* the window (e.g. `seed_start`), so some buckets
  land on each side. Watch step 12 for this; note the resolution in the PBI.
- **iqr_bounds.json write target** — must be `data_exp/models/`, never
  production `models/`. The override sets `model.store_path`; verify in step 12
  the file materializes under the experiment root and production `models/` is
  untouched.
- **`_feature` reads TWO XComs** — `bronze_path` from `ingest` AND `window_end`
  from `read_watermark`. EXP-03a's `_validate` read only the former. The extra
  pull is needed for the OQ4 cutoff; noted so the reviewer sees why `_feature`
  differs structurally from `_validate`.
- **Silver-before-gold ordering** — `run_feature` MUST call the cleaner before
  the engineer (the engineer reads the silver the cleaner writes). The
  `test_cleaner_runs_before_engineer` test guards this; a refactor that
  reorders them would fail it.
- **Container-only smoke for the gold-shape ACs** — the "one row per
  `(zone, bucket)`" and "required columns present" ACs need a real
  shapefile + real bronze, so they're verified in-container (step 12), not in
  host pytest. Host tests prove the wiring; the container proves the artifact.
- **Blast radius** — confined to `orchestration_experiment/` + two new test
  files + one compose mount + two lines in `requirements-exp.txt`. No change to
  `app/`, `orchestration/`, `implementation_checklist/`, `config.yml`, or
  production `data/`.

---

## Open questions to resolve during build (from PBI EXP-03b §8)

- **OQ1** → inject config dict (locked). Test signature commits to it.
- **OQ2** → manual causality spot-check in the note (locked); no new leakage
  test — the property is covered by `tests/test_feature.py`.
- **OQ3** → let `iqr_bounds.json` write to `data_exp/models/` (locked); verified
  in step 12 not to touch production `models/`.
- **OQ4** → override `inference.simulation.start_hour` to `window_end` (locked);
  step 12 confirms the all-train path survives, with a documented fallback
  (cutoff inside the window) if it doesn't.
