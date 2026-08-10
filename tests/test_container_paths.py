"""Tests for orchestration/lib/container_paths.py.

Step 7 (hardcoded-values audit) — verifies that container_paths(config) produces
the exact /opt/airflow/... strings that the DAG previously hardcoded as module-level
constants. The helper is Airflow-free and fully host-testable.

The equality assertion doubles as a regression guard: if a config key changes, the
DAG string changes, and this test catches the divergence before a container run.
"""

from __future__ import annotations

import pytest

from app.core.config import AppConfig
from orchestration.lib.container_paths import container_paths
from tests.conftest import _FULL_CONFIG_DICT

_CONFIG = AppConfig.model_validate(_FULL_CONFIG_DICT)

_EXPECTED = {
    "raw": "/opt/airflow/data/raw/yellow_tripdata",
    "bronze": "/opt/airflow/data/bronze/yellow_tripdata/bronze.parquet",
    "silver": "/opt/airflow/data/silver/yellow_tripdata_cleaned.parquet",
    "gold": "/opt/airflow/data/gold/yellow_tripdata",
    "shapefile": "/opt/airflow/data/reference/taxi_zones.shp",
    "models": "/opt/airflow/data/models",
    "predictions": "/opt/airflow/data/predictions/demand_predictions.parquet",
    "monitoring": "/opt/airflow/data/monitoring/predictions",
    "watermark": "/opt/airflow/data/watermarks/refresh_15min",
}


class TestContainerPaths:
    """container_paths() reproduces today's 9 hardcoded DAG literals."""

    def test_all_paths_equal_current_dag_literals(self) -> None:
        """container_paths(config) reproduces all 9 /opt/airflow/... strings."""
        # Act
        result = container_paths(_CONFIG)

        # Assert
        assert result == _EXPECTED

    @pytest.mark.parametrize("key,expected", list(_EXPECTED.items()))
    def test_individual_path_matches_dag_literal(self, key: str, expected: str) -> None:
        """Each key individually matches the corresponding DAG literal."""
        # Act
        result = container_paths(_CONFIG)

        # Assert
        assert result[key] == expected

    def test_custom_prefix_replaces_opt_airflow(self) -> None:
        """A custom prefix is used instead of /opt/airflow."""
        # Act
        result = container_paths(_CONFIG, prefix="/custom/mount")

        # Assert
        assert result["raw"] == "/custom/mount/data/raw/yellow_tripdata"
        assert result["watermark"] == "/custom/mount/data/watermarks/refresh_15min"

    def test_models_path_under_prefix(self) -> None:
        """container_paths(cfg)['models'] starts with /opt/airflow (PBI 37 Step 7)."""
        # Arrange / Act
        result = container_paths(_CONFIG)

        # Assert
        assert result["models"].startswith("/opt/airflow")

    @pytest.mark.integration
    def test_train_infer_model_paths_agree(self) -> None:
        """build_train_config and build_predict_config yield equal model.store_path (PBI 37 Step 7)."""
        from orchestration.lib.predict_helpers import build_predict_config
        from orchestration.lib.training_helpers import build_train_config

        base_cfg = _CONFIG
        paths = container_paths(base_cfg)
        model_root = paths["models"]

        train_cfg = build_train_config(base_cfg, model_root=model_root)
        predict_cfg = build_predict_config(
            base_cfg,
            gold_root=paths["gold"],
            model_root=model_root,
            predictions_path=paths["predictions"],
        )

        assert train_cfg.model.store_path == predict_cfg.model.store_path

    def test_leading_slash_in_config_path_raises_or_warns(self) -> None:
        """A leading slash in a config path would silently drop the prefix.

        This test documents the foot-gun: pathlib silently replaces the prefix
        when the relative path starts with '/'. We verify the helper rejects it.
        """
        bad_config = _CONFIG.model_copy(
            update={
                "data": _CONFIG.data.model_copy(update={"raw_path": "/absolute/path"})
            }
        )

        # Assert — leading slash raises ValueError
        with pytest.raises(ValueError, match="leading slash"):
            container_paths(bad_config)
