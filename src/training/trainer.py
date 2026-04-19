"""Model training for NYC Taxi demand forecasting."""

from __future__ import annotations

import os
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from src.shared.config import load_config

MODELS: dict[str, type[LGBMRegressor]] = {
    "LGBMRegressor": LGBMRegressor,
}

TARGET_COL = "trip_count"


class Trainer:
    """Trains a demand prediction model using the Gold layer data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise Trainer with configuration.

        Args:
            config: Parsed configuration dictionary. If ``None``, loads
                from ``config.yml`` via :func:`src.shared.config.load_config`.
        """
        self.config = config or load_config()
        model_cfg: dict[str, Any] = self.config["model"]
        self.model_name: str = model_cfg["name"]
        self.store_path: str = model_cfg["store_path"]
        self.model: LGBMRegressor = MODELS[self.model_name](
            **model_cfg.get("params", {})
        )

    def feature_target_separator(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series[Any]]:
        """Split features and target from the dataset.

        Args:
            data: DataFrame containing both features and the target column.

        Returns:
            A ``(X, y)`` tuple of features DataFrame and target Series.
        """
        x = data.drop(columns=[TARGET_COL])
        y = data[TARGET_COL]
        return x, y

    def train_model(
        self, x_train: pd.DataFrame, y_train: pd.Series[Any]
    ) -> None:
        """Fit the model on training data.

        Args:
            x_train: Training feature matrix.
            y_train: Training target values.
        """
        self.model.fit(x_train, y_train)

    def save_model(self) -> None:
        """Save the trained model to the configured store path."""
        os.makedirs(self.store_path, exist_ok=True)
        joblib.dump(self.model, os.path.join(self.store_path, "model.pkl"))
