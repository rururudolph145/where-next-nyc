"""Path helpers wrapping the ``data.*`` section of ``config.yml``.

Thin accessor functions so call sites don't repeat attribute lookups.
"""

from pathlib import Path

from app.core.config import AppConfig


def bronze_path(config: AppConfig) -> Path:
    """Return the configured Bronze-layer parquet path."""
    return Path(config.data.bronze_path)


def silver_path(config: AppConfig) -> Path:
    """Return the configured Silver-layer parquet path."""
    return Path(config.data.silver_path)


def gold_path(config: AppConfig) -> Path:
    """Return the configured Gold-layer parquet directory."""
    return Path(config.data.gold_path)


def raw_path(config: AppConfig) -> Path:
    """Return the configured raw-data directory."""
    return Path(config.data.raw_path)


def model_store_path(config: AppConfig) -> Path:
    """Return the configured directory where trained models are saved."""
    return Path(config.model.store_path)


def validation_log_dir(config: AppConfig) -> Path:
    """Return the directory where validation log JSON files are written.

    Resolves to the ``_validation`` subdirectory that sits beside the Silver
    parquet file (i.e. ``<silver_path.parent>/_validation``).
    """
    return Path(config.data.silver_path).parent / "_validation"
