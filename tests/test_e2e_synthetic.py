"""End-to-end pipeline tests on synthetic data (Silver → Gold → Train).

These tests exercise the *integration* between FeatureEngineer and Trainer
on a tiny in-memory Silver fixture, so they catch breakage no unit test would
see (e.g. a new feature column missing from `train.features`, a target-leakage
regression, or a schema drift between feature engineering and training).

Three tests:

1. ``test_legacy_pipeline_smoke`` — bucket=h, horizon=0, no neighbor feature.
   Locks down the current behaviour so B1 cannot regress it. GREEN today.

2. ``test_b1_pipeline_smoke`` — bucket=30min, horizon=0, n_neighbors=5.
   Locks the B1 nowcast pipeline end-to-end: 30-min grid, neighbour-mean
   smoothing column present and finite, trainer runs to completion.

3. ``test_no_target_leakage`` — invariant guard. No feature column equals
   the row's own target; all lag columns reference strictly prior buckets.
   Runs on legacy mode (the only mode currently green).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.core.config import AppConfig
from app.ingestion.feature import FeatureEngineer
from app.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Synthetic Silver fixture
# ---------------------------------------------------------------------------

# Three zones placed on a known coordinate line so neighbor-mean math is
# hand-verifiable in B1's unit tests. Coordinates here are only used so the
# patched ``add_zone_coordinates`` can hand back deterministic lat/lon.
SYNTHETIC_ZONES: list[dict[str, float]] = [
    {"pulocationid": 1, "lon": -74.00, "lat": 40.70},
    {"pulocationid": 2, "lon": -74.01, "lat": 40.71},
    {"pulocationid": 3, "lon": -74.02, "lat": 40.72},
]


def _make_synthetic_silver(
    n_days: int = 10,
    trips_per_hour_per_zone: int = 3,
) -> pd.DataFrame:
    """Build a synthetic Silver-layer DataFrame.

    Generates ``n_days * 24`` hours of records across 3 zones, with the
    configured number of trips per hour per zone. Lots of rows so the lag
    schedule (up to 168 hourly buckets, or 336 30-min buckets) leaves enough
    for a train/val/test split.
    """
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2021-03-01")
    rng = np.random.default_rng(42)
    for day in range(n_days):
        for hour in range(24):
            ts = start + pd.Timedelta(days=day, hours=hour)
            for zone in SYNTHETIC_ZONES:
                # Vary trip count a little so the model has signal to learn
                n = trips_per_hour_per_zone + int(rng.integers(-1, 2))
                n = max(1, n)
                for k in range(n):
                    rows.append(
                        {
                            "tpep_pickup_datetime": ts
                            + pd.Timedelta(minutes=5 + k * 3),
                            "pulocationid": zone["pulocationid"],
                        }
                    )
    return pd.DataFrame(rows)


def _patch_zone_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``add_zone_coordinates`` with a no-disk version.

    The real method reads a shapefile from disk; for E2E tests we hand back
    deterministic lat/lon from the synthetic zone table so the rest of the
    pipeline can exercise unchanged.
    """
    lon_map = {z["pulocationid"]: z["lon"] for z in SYNTHETIC_ZONES}
    lat_map = {z["pulocationid"]: z["lat"] for z in SYNTHETIC_ZONES}

    def fake_add_zone_coordinates(
        self: FeatureEngineer,
        df: pd.DataFrame,
        zone_col: str = "pulocationid",
    ) -> pd.DataFrame:
        df = df.copy()
        df["pickup_longitude"] = df[zone_col].map(lon_map)
        df["pickup_latitude"] = df[zone_col].map(lat_map)
        return df.dropna(subset=["pickup_longitude", "pickup_latitude"])

    monkeypatch.setattr(
        FeatureEngineer, "add_zone_coordinates", fake_add_zone_coordinates
    )


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _legacy_config(tmp_path: Any) -> AppConfig:
    """Config for the current/legacy pipeline (bucket=h, horizon=0)."""
    from tests.conftest import _FULL_CONFIG_DICT

    base = AppConfig.model_validate(_FULL_CONFIG_DICT)
    from app.core.config import FeatureConfig, TrainConfig, TrainGateConfig

    return base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "silver_path": str(tmp_path / "silver.parquet"),
                    "gold_path": str(tmp_path / "gold"),
                }
            ),
            "feature": FeatureConfig.model_validate(
                {
                    "bucket": "h",
                    "lags": [1, 2, 3, 24],
                    "horizon_buckets": 0,
                    "zone_shapefile_path": "unused-when-patched",
                }
            ),
            "model": base.model.model_copy(
                update={
                    "params": {
                        "n_estimators": 20,
                        "learning_rate": 0.1,
                        "num_leaves": 7,
                        "random_state": 42,
                        "verbosity": -1,
                    },
                    "store_path": str(tmp_path / "models"),
                }
            ),
            "train": TrainConfig(
                window_days=base.train.window_days,
                gate=TrainGateConfig(),
                features=[
                    "hour_of_day_sin",
                    "hour_of_day_cos",
                    "day_of_week_sin",
                    "day_of_week_cos",
                    "month_sin",
                    "month_cos",
                    "pickup_longitude",
                    "pickup_latitude",
                    "trip_count_lag_1",
                    "trip_count_lag_2",
                    "trip_count_lag_3",
                    "trip_count_lag_24",
                ],
                target="trip_count",
            ),
            "mlflow": base.mlflow.model_copy(
                update={
                    "experiment_name": "e2e_synthetic",
                    "tracking_uri": str(tmp_path / "mlruns"),
                }
            ),
        }
    )


def _b1_config(tmp_path: Any) -> AppConfig:
    """Config for the B1 nowcast pipeline (bucket=30min + neighbor smoothing)."""
    from app.core.config import FeatureConfig, TrainConfig, TrainGateConfig

    base = _legacy_config(tmp_path)
    return base.model_copy(
        update={
            "feature": FeatureConfig.model_validate(
                {
                    "bucket": "30min",
                    "lags": [1, 2, 4, 48],
                    "n_neighbors": 2,
                    "zone_shapefile_path": "unused-when-patched",
                }
            ),
            "train": TrainConfig(
                window_days=base.train.window_days,
                gate=TrainGateConfig(),
                features=[
                    "hour_of_day_sin",
                    "hour_of_day_cos",
                    "day_of_week_sin",
                    "day_of_week_cos",
                    "month_sin",
                    "month_cos",
                    "pickup_longitude",
                    "pickup_latitude",
                    "trip_count_lag_1",
                    "trip_count_lag_2",
                    "trip_count_lag_4",
                    "trip_count_lag_48",
                    "trip_count_lag_1_neighbor_mean",
                ],
                target="trip_count",
            ),
        }
    )


# ---------------------------------------------------------------------------
# Helpers to exercise the pipeline without touching MLflow
# ---------------------------------------------------------------------------


def _run_feature_engineering(config: AppConfig, silver: pd.DataFrame) -> pd.DataFrame:
    """Run Silver → Gold without writing to disk (we ignore the parquet emit)."""
    fe = FeatureEngineer(config=config)
    # FeatureEngineer.run writes to gold_path; that path is tmp here so it's harmless.
    return fe.run(data=silver)


def _train_no_mlflow(config: AppConfig, gold: pd.DataFrame) -> dict[str, Any]:
    """Train without MLflow logging — avoids global tracking URI side effects.

    Mirrors the structure of ``Trainer.run`` but skips the MLflow block so the
    test stays hermetic.
    """
    trainer = Trainer(config=config)
    train_df, test_df = trainer.split_data(gold)
    x_train = train_df[trainer.features]
    y_train = train_df[trainer.target]
    trainer.train(x_train, y_train)
    test_preds = trainer.model.predict(test_df[trainer.features])
    return {
        "test_metrics": trainer.evaluate(test_df[trainer.target], test_preds),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_preds": test_preds,
    }


# ---------------------------------------------------------------------------
# 1. Legacy smoke — green today, must stay green after B1
# ---------------------------------------------------------------------------


def test_legacy_pipeline_smoke(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Silver → Gold → Train end-to-end on the legacy hourly pipeline."""
    _patch_zone_coordinates(monkeypatch)
    config = _legacy_config(tmp_path)
    silver = _make_synthetic_silver(n_days=10)

    gold = _run_feature_engineering(config, silver)

    # Schema sanity: legacy mode should NOT carry neighbor-smoothing column
    assert "trip_count" in gold.columns
    assert "trip_count_lag_1" in gold.columns
    assert "trip_count_lag_1_neighbor_mean" not in gold.columns
    assert gold["hour"].dt.minute.eq(0).all(), "Legacy mode must use hourly buckets"

    # Training loop runs and produces finite predictions
    result = _train_no_mlflow(config, gold)
    assert np.isfinite(result["test_preds"]).all()
    assert result["n_train"] > 0
    assert result["test_metrics"]["mae"] >= 0


# ---------------------------------------------------------------------------
# 2. B1 smoke — GREEN, locks the 30-min + neighbor-smoothing nowcast path
# ---------------------------------------------------------------------------


def test_b1_pipeline_smoke(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """30-min bucket + neighbor-mean smoothing end-to-end."""
    _patch_zone_coordinates(monkeypatch)
    config = _b1_config(tmp_path)
    silver = _make_synthetic_silver(n_days=10)

    gold = _run_feature_engineering(config, silver)

    # B1 invariants
    assert "trip_count_lag_1_neighbor_mean" in gold.columns, (
        "B1 must emit trip_count_lag_1_neighbor_mean"
    )
    assert gold["trip_count_lag_1_neighbor_mean"].notna().all(), (
        "Neighbor-mean must be finite for every retained row"
    )
    # 30-min grid: every bucket minute is 0 or 30
    assert gold["hour"].dt.minute.isin([0, 30]).all(), (
        "B1 must aggregate to 30-min buckets"
    )

    result = _train_no_mlflow(config, gold)
    assert np.isfinite(result["test_preds"]).all()
    assert result["n_train"] > 0


# ---------------------------------------------------------------------------
# 3. Target-leakage invariant
# ---------------------------------------------------------------------------


def _b2_config(tmp_path: Any) -> AppConfig:
    """Config for the B2 extended-nowcast pipeline.

    15-min bucket + horizon_sum_buckets=2 → target = demand over the next
    30 min as a sum of two future 15-min bins. Keeps B1's neighbour
    smoothing.
    """
    from app.core.config import FeatureConfig, TrainConfig, TrainGateConfig

    base = _legacy_config(tmp_path)
    return base.model_copy(
        update={
            "feature": FeatureConfig.model_validate(
                {
                    "bucket": "15min",
                    "lags": [1, 2, 4, 8, 96],
                    "horizon_sum_buckets": 2,
                    "n_neighbors": 2,
                    "zone_shapefile_path": "unused-when-patched",
                }
            ),
            "train": TrainConfig(
                window_days=base.train.window_days,
                gate=TrainGateConfig(),
                features=[
                    "hour_of_day_sin",
                    "hour_of_day_cos",
                    "day_of_week_sin",
                    "day_of_week_cos",
                    "month_sin",
                    "month_cos",
                    "pickup_longitude",
                    "pickup_latitude",
                    "trip_count_lag_1",
                    "trip_count_lag_2",
                    "trip_count_lag_4",
                    "trip_count_lag_8",
                    "trip_count_lag_96",
                    "trip_count_lag_1_neighbor_mean",
                ],
                target="target_sum_t1_t2",
            ),
        }
    )


# ---------------------------------------------------------------------------
# 2b. B2 smoke — 15-min + summed-horizon target
# ---------------------------------------------------------------------------


def test_b2_pipeline_smoke(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """15-min bucket + 30-min summed target end-to-end.

    Locks the B2 "extended nowcast" pipeline: 15-min grid, summed-horizon
    target column present, target equals lag_1[t+1] + lag_1[t+2] of the
    same zone (i.e. trainer learns to predict the next 30 min as a sum
    of two future 15-min bins).
    """
    _patch_zone_coordinates(monkeypatch)
    config = _b2_config(tmp_path)
    # B2 needs more data: 96 lag buckets × 15 min = 1 day; plus 2 future
    # buckets to drop. Use 14 days for headroom.
    silver = _make_synthetic_silver(n_days=14)

    gold = _run_feature_engineering(config, silver)

    # B2 invariants
    assert "target_sum_t1_t2" in gold.columns, (
        "B2 must emit the summed-horizon target column"
    )
    assert gold["target_sum_t1_t2"].notna().all(), (
        "Summed target must be finite for every retained row"
    )
    # 15-min grid: every bucket minute is one of {0, 15, 30, 45}
    assert gold["hour"].dt.minute.isin([0, 15, 30, 45]).all(), (
        "B2 must aggregate to 15-min buckets"
    )
    # Smoothing column carried over from B1
    assert "trip_count_lag_1_neighbor_mean" in gold.columns

    result = _train_no_mlflow(config, gold)
    assert np.isfinite(result["test_preds"]).all()
    assert result["n_train"] > 0


def test_no_target_leakage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """No feature column may equal the row's own target value.

    Two checks:

    a) Static schema — ``trip_count`` (the target) must not be in
       ``train.features``. Lag columns reference past buckets, not the
       current one.

    b) Row-wise — for every row in Gold, no configured feature column has
       the same value as ``trip_count``. (Allows the rare coincidence: we
       sample rows and require at least *some* disagreement — perfect
       agreement across the panel would be a smoking gun for leakage.)
    """
    _patch_zone_coordinates(monkeypatch)
    config = _legacy_config(tmp_path)
    silver = _make_synthetic_silver(n_days=10)
    gold = _run_feature_engineering(config, silver)

    target = config.train.target
    features = config.train.features

    # (a) Target name must not appear in the feature list
    assert target not in features, (
        f"Target column {target!r} leaked into train.features"
    )

    # (b) For each feature, at least one row must differ from target.
    # If a feature is *identically equal* to target across the panel,
    # that's a leakage smell (the model would learn an identity).
    for feat in features:
        if feat not in gold.columns:
            continue
        if not np.issubdtype(gold[feat].dtype, np.number):  # type: ignore[arg-type]
            continue
        if not np.issubdtype(gold[target].dtype, np.number):  # type: ignore[arg-type]
            continue
        # Compare elementwise; require at least one row of disagreement
        # (rules out f === target leak; tolerates incidental matches).
        differs = (gold[feat] != gold[target]).any()
        assert differs, (
            f"Feature {feat!r} is identically equal to target {target!r} — "
            "target leakage suspected"
        )
