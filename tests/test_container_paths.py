"""Tests for orchestration/lib/container_paths.py.

Step 7 (hardcoded-values audit) — verifies that container_paths(config) produces
the exact /opt/airflow/... strings that the DAG previously hardcoded as module-level
constants. The helper is Airflow-free and fully host-testable.

The equality assertion doubles as a regression guard: if a config key changes, the
DAG string changes, and this test catches the divergence before a container run.
"""

from __future__ import annotations

import pytest

from orchestration.lib.container_paths import container_paths


_CONFIG = {
    "data": {
        "raw_path": "data/raw/yellow_tripdata",
        "bronze_path": "data/bronze/yellow_tripdata/bronze.parquet",
        "silver_path": "data/silver/yellow_tripdata_cleaned.parquet",
        "gold_path": "data/gold/yellow_tripdata",
    },
    "feature": {
        "zone_shapefile_path": "data/reference/taxi_zones.shp",
    },
    "model": {
        "store_path": "data/models",
    },
    "serving": {
        "predictions_path": "data/predictions/demand_predictions.parquet",
        "watermark_path": "data/watermarks/refresh_15min",
    },
    "monitoring": {
        "predictions_path": "data/monitoring/predictions",
    },
}

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

    def test_leading_slash_in_config_path_raises_or_warns(self) -> None:
        """A leading slash in a config path would silently drop the prefix.

        This test documents the foot-gun: pathlib silently replaces the prefix
        when the relative path starts with '/'. We verify the helper rejects it.
        """
        bad_config = {
            **_CONFIG,
            "data": {
                **_CONFIG["data"],
                "raw_path": "/absolute/path",  # starts with /
            },
        }

        # Assert — leading slash raises ValueError
        with pytest.raises(ValueError, match="leading slash"):
            container_paths(bad_config)
