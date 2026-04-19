"""Model inference and evaluation for NYC Taxi demand forecasting."""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shared.config import load_config

TARGET_COL = "trip_count"


class Predictor:
    """Loads a trained model and generates predictions with evaluation metrics."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise Predictor with configuration.

        Args:
            config: Parsed configuration dictionary. If ``None``, loads
                from ``config.yml`` via :func:`src.shared.config.load_config`.
        """
        self.config = config or load_config()
        self.model: Any = self.load_model()

    def load_model(self) -> Any:
        """Load the trained model from the configured store path.

        Returns:
            The deserialized model object.
        """
        store_path: str = self.config["model"]["store_path"]
        return joblib.load(os.path.join(store_path, "model.pkl"))

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

    def evaluate_model(
        self, x_test: pd.DataFrame, y_test: pd.Series[Any]
    ) -> tuple[float, float, float]:
        """Generate predictions and compute evaluation metrics.

        Args:
            x_test: Test feature matrix.
            y_test: True target values.

        Returns:
            A ``(mae, rmse, r2)`` tuple of evaluation metrics.
        """
        preds: np.ndarray[Any, np.dtype[np.floating[Any]]] = self.model.predict(x_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        return mae, rmse, r2
