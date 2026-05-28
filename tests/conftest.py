"""Shared pytest fixtures and helpers for all test modules."""

from __future__ import annotations

from unittest.mock import MagicMock


def make_mock_version(
    version: str = "3",
    stage: str = "Production",
    creation_timestamp_ms: int = 1_710_058_921_000,
    tags: dict[str, str] | None = None,
) -> MagicMock:
    """Return a MagicMock simulating an MLflow ModelVersion.

    Shared across test modules that exercise MlflowClient.get_latest_versions().

    Args:
        version: The model version string (e.g. "3").
        stage: The MLflow stage string ("Production" or "Archived").
        creation_timestamp_ms: Unix epoch milliseconds for the version's creation time.
        tags: Optional dict of MLflow model version tags.

    Returns:
        MagicMock with ``.version``, ``.current_stage``, ``.creation_timestamp``,
        and ``.tags`` attributes set.
    """
    mv = MagicMock()
    mv.version = version
    mv.current_stage = stage
    mv.creation_timestamp = creation_timestamp_ms
    mv.tags = tags if tags is not None else {}
    return mv
