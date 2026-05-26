"""Batch inference for NYC Taxi demand forecasting.

Loads a trained model, generates predictions on Gold layer data,
builds a predictions table, evaluates metrics, and saves results.
"""

from __future__ import annotations

import os
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from app.core.config import load_config
from app.core.metrics import evaluate as _evaluate


class Predictor:
    """Loads a trained model and generates batch predictions."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise Predictor with configuration.

        Args:
            config: Parsed configuration dictionary. If ``None``, loads
                from ``config.yml`` via :func:`app.core.config.load_config`.
        """
        self.config = config or load_config()

        model_cfg: dict[str, Any] = self.config["model"]
        self.store_path: str = model_cfg["store_path"]

        train_cfg: dict[str, Any] = self.config["train"]
        self.features: list[str] = train_cfg["features"]
        self.target: str = train_cfg["target"]

        self.gold_path: str = self.config["data"]["gold_path"]

        inference_cfg: dict[str, Any] = self.config.get("inference", {})
        self.output_path: str = inference_cfg.get(
            "output_path", "data/predictions/demand_predictions.parquet"
        )

        self.model: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> Any:
        """Load the trained model from the configured store path.

        Returns:
            The deserialized model object.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = os.path.join(self.store_path, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)
        return self.model

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_gold(self) -> pd.DataFrame:
        """Load the Gold layer parquet.

        Returns:
            Gold layer DataFrame with features and target.
        """
        gold_file = os.path.join(self.gold_path, "hourly_demand_featured.parquet")
        return pd.read_parquet(gold_file)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Run the model on the feature columns of ``df``.

        Args:
            df: DataFrame containing at least the configured feature columns.

        Returns:
            NumPy array of predicted values.
        """
        x = df[self.features]
        preds: np.ndarray[Any, np.dtype[np.floating[Any]]] = self.model.predict(x)
        return preds

    # ------------------------------------------------------------------
    # Predictions table
    # ------------------------------------------------------------------

    def build_predictions_table(
        self,
        df: pd.DataFrame,
        preds: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ) -> pd.DataFrame:
        """Build a predictions table from input data and predictions.

        Args:
            df: Gold layer DataFrame with ``pulocationid``, ``hour``,
                and the target column.
            preds: Predicted values from :meth:`predict`.

        Returns:
            DataFrame with columns ``pulocationid``, ``hour``,
            ``predicted_trip_count``, ``actual_trip_count``.
        """
        return pd.DataFrame(
            {
                "pulocationid": df["pulocationid"].values,
                "hour": df["hour"].values,
                "predicted_trip_count": preds,
                "actual_trip_count": df[self.target].values,
            }
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        y_true: pd.Series[Any],
        y_pred: np.ndarray[Any, np.dtype[Any]],
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Delegates to :func:`app.core.metrics.evaluate`.

        Args:
            y_true: True target values.
            y_pred: Predicted values.

        Returns:
            Dictionary with keys ``mae``, ``rmse``, ``mape``, ``wape``.
        """
        return _evaluate(y_true, y_pred)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_predictions(self, predictions_df: pd.DataFrame) -> str:
        """Save the predictions table to a parquet file.

        Args:
            predictions_df: DataFrame produced by :meth:`build_predictions_table`.

        Returns:
            Path to the saved parquet file.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        predictions_df.to_parquet(self.output_path, index=False)
        return self.output_path

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame | None = None) -> dict[str, Any]:
        """Execute the full inference pipeline.

        1. Load model.
        2. Load Gold data (or use provided DataFrame).
        3. Generate predictions.
        4. Build predictions table.
        5. Evaluate metrics.
        6. Save predictions.

        Args:
            data: Optional Gold layer DataFrame. If ``None``, loads from
                the configured gold path.

        Returns:
            Dictionary with ``metrics``, ``output_path``, and
            ``n_predictions``.
        """
        self.load_model()

        if data is None:
            data = self.load_gold()

        preds = self.predict(data)
        predictions_df = self.build_predictions_table(data, preds)

        y_true: pd.Series[Any] = data[self.target]
        metrics = self.evaluate(y_true, preds)

        print(
            f"Inference: MAE={metrics['mae']:.2f}  "
            f"RMSE={metrics['rmse']:.2f}  "
            f"MAPE={metrics['mape']:.2f}%  "
            f"WAPE={metrics['wape']:.2f}%"
        )

        output_path = self.save_predictions(predictions_df)
        print(f"Predictions saved → {output_path} ({len(predictions_df):,} rows)")

        return {
            "metrics": metrics,
            "output_path": output_path,
            "n_predictions": len(predictions_df),
        }
