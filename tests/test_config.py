"""Tests for the typed AppConfig model and load_config()."""

from __future__ import annotations

import pytest
import yaml

from app.core.config import AppConfig, load_config


class TestLoadConfigReturnsAppConfig:
    """AC: load_config() returns a typed AppConfig, not a raw dict."""

    def test_load_config_returns_appconfig_instance(self) -> None:
        """load_config() must return an AppConfig, not a dict."""
        # Act
        result = load_config()
        # Assert
        assert isinstance(result, AppConfig)


class TestAppConfigModelValidate:
    """AC: AppConfig.model_validate succeeds on a well-formed config.yml."""

    def test_validates_full_config_yml(self) -> None:
        """model_validate against real config.yml must succeed and return AppConfig."""
        # Arrange
        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        # Act
        cfg = AppConfig.model_validate(raw)
        # Assert
        assert isinstance(cfg, AppConfig)

    def test_spot_check_clean_iqr_multiplier(self) -> None:
        """clean.iqr_multiplier equals config.yml value 1.5."""
        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        cfg = AppConfig.model_validate(raw)
        assert cfg.clean.iqr_multiplier == 1.5

    def test_spot_check_ingest_zone_id_min(self) -> None:
        """ingest.zone_id_min equals config.yml value 1."""
        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        cfg = AppConfig.model_validate(raw)
        assert cfg.ingest.zone_id_min == 1

    def test_spot_check_feature_horizon_sum_buckets(self) -> None:
        """feature.horizon_sum_buckets equals config.yml value 2 (overrides code default 0)."""
        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        cfg = AppConfig.model_validate(raw)
        assert cfg.feature.horizon_sum_buckets == 2


class TestAppConfigRequired:
    """AC: missing required fields raise ValidationError, not KeyError."""

    def test_missing_data_raw_path_raises(self) -> None:
        """Omitting data.raw_path (a required field) must raise ValidationError."""
        from pydantic import ValidationError

        # Arrange — build a near-complete dict missing data.raw_path
        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        del raw["data"]["raw_path"]
        # Act / Assert
        with pytest.raises(ValidationError):
            AppConfig.model_validate(raw)

    def test_missing_top_level_section_raises(self) -> None:
        """Omitting an entire required section (e.g. data) raises ValidationError."""
        from pydantic import ValidationError

        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        del raw["data"]
        with pytest.raises(ValidationError):
            AppConfig.model_validate(raw)


class TestAppConfigTypeMismatch:
    """AC: type-wrong values raise ValidationError at load, not at use-time."""

    def test_iqr_multiplier_string_raises(self) -> None:
        """clean.iqr_multiplier='x' (non-numeric string) must raise ValidationError."""
        from pydantic import ValidationError

        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        raw["clean"]["iqr_multiplier"] = "x"
        with pytest.raises(ValidationError):
            AppConfig.model_validate(raw)


class TestFeatureLagsAlias:
    """AC: lags/lag_hours AliasChoices resolves correctly."""

    def test_lags_key_resolves(self) -> None:
        """Primary key 'lags' is accepted and values are parsed."""
        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        raw["feature"]["lags"] = [1, 2, 4]
        cfg = AppConfig.model_validate(raw)
        assert cfg.feature.lags == [1, 2, 4]

    def test_lag_hours_alias_resolves(self) -> None:
        """Legacy alias 'lag_hours' is accepted when 'lags' is absent."""
        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        raw["feature"]["lag_hours"] = [8, 24]
        del raw["feature"]["lags"]
        cfg = AppConfig.model_validate(raw)
        assert cfg.feature.lags == [8, 24]

    def test_lags_wins_over_lag_hours(self) -> None:
        """When both keys present, 'lags' takes precedence."""
        with open("config.yml") as f:
            raw = yaml.safe_load(f)
        raw["feature"]["lags"] = [1, 2]
        raw["feature"]["lag_hours"] = [99, 100]
        cfg = AppConfig.model_validate(raw)
        assert cfg.feature.lags == [1, 2]
