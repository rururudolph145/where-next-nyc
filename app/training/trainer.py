"""Model training for NYC Taxi demand forecasting.

Loads Gold layer data, splits by time, trains LGBMRegressor,
evaluates on val/test, logs to MLflow, and saves the model artifact.
"""

from __future__ import annotations

import os
from typing import Any

import joblib  # type: ignore[import-untyped]
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from app.core.config import load_config

# Re-exported so legacy ``patch("src.training.train.mlflow")`` paths keep
# working via the src.training.train shim. Listed in __all__ to satisfy
# mypy --strict's implicit-reexport check.
__all__ = ["MODELS", "Trainer", "mlflow"]

MODELS: dict[str, type[LGBMRegressor]] = {
    "LGBMRegressor": LGBMRegressor,
}


class Trainer:
    """Trains a demand prediction model using the Gold layer data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise Trainer with configuration.

        Args:
            config: Parsed configuration dictionary. If ``None``, loads
                from ``config.yml`` via :func:`app.core.config.load_config`.
        """
        self.config = config or load_config()

        model_cfg: dict[str, Any] = self.config["model"]
        self.model_name: str = model_cfg["name"]
        self.model_params: dict[str, Any] = model_cfg.get("params", {})
        self.store_path: str = model_cfg["store_path"]
        self.model: LGBMRegressor = MODELS[self.model_name](**self.model_params)

        train_cfg: dict[str, Any] = self.config["train"]
        self.features: list[str] = train_cfg["features"]
        self.target: str = train_cfg["target"]
        self.split_ratios: list[float] = train_cfg["split_ratios"]

        self.gold_path: str = self.config["data"]["gold_path"]

        mlflow_cfg: dict[str, Any] = self.config.get("mlflow", {})
        self.experiment_name: str = mlflow_cfg.get(
            "experiment_name", "ny_taxi_demand_model"
        )
        self.tracking_uri: str = mlflow_cfg.get("tracking_uri", "mlruns")

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
    # Time-based splitting
    # ------------------------------------------------------------------

    def split_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test using strict time ordering.

        The data is sorted by ``hour`` and split at the 80th and 90th
        percentile boundaries. No shuffling is performed.

        Args:
            df: Gold layer DataFrame containing an ``hour`` column.

        Returns:
            A ``(train, val, test)`` tuple of DataFrames.
        """
        df = df.sort_values("hour").reset_index(drop=True)

        n = len(df)
        train_end = int(n * self.split_ratios[0])
        val_end = int(n * (self.split_ratios[0] + self.split_ratios[1]))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        return train, val, test

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, x_train: pd.DataFrame, y_train: pd.Series[Any]) -> None:
        """Fit the model on training data.

        Args:
            x_train: Training feature matrix.
            y_train: Training target values.
        """
        self.model.fit(x_train, y_train)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        y_true: pd.Series[Any],
        y_pred: np.ndarray[Any, np.dtype[Any]],
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            y_true: True target values.
            y_pred: Predicted values.

        Returns:
            Dictionary with keys ``mae``, ``rmse``, ``mape``, ``wape``.
        """
        actual = np.asarray(y_true.values)
        mae = float(np.mean(np.abs(actual - y_pred)))
        rmse = float(np.sqrt(np.mean((actual - y_pred) ** 2)))

        mask = actual != 0
        if bool(mask.any()):
            mape = float(
                np.mean(np.abs((actual[mask] - y_pred[mask]) / actual[mask])) * 100
            )
        else:
            mape = 0.0

        total = float(np.sum(np.abs(actual)))
        wape = (
            float(np.sum(np.abs(actual - y_pred)) / total * 100) if total > 0 else 0.0
        )

        return {"mae": mae, "rmse": rmse, "mape": mape, "wape": wape}

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self) -> str:
        """Save the trained model to the configured store path.

        Returns:
            Path to the saved model file.
        """
        os.makedirs(self.store_path, exist_ok=True)
        model_path = os.path.join(self.store_path, "model.pkl")
        joblib.dump(self.model, model_path)
        return model_path

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame | None = None) -> dict[str, Any]:
        """Execute the full training pipeline.

        1. Load Gold data (or use provided DataFrame).
        2. Split into train / val / test by time.
        3. Train LGBMRegressor on training set.
        4. Evaluate on val and test sets.
        5. Log params, metrics, and model to MLflow.
        6. Save model artifact to disk.

        Args:
            data: Optional Gold layer DataFrame. If ``None``, loads from
                the configured gold path.

        Returns:
            Dictionary with ``val_metrics``, ``test_metrics``, and
            ``model_path``.
        """
        if data is None:
            data = self.load_gold()

        train_df, val_df, test_df = self.split_data(data)

        x_train = train_df[self.features]
        y_train = train_df[self.target]
        x_val = val_df[self.features]
        y_val = val_df[self.target]
        x_test = test_df[self.features]
        y_test = test_df[self.target]

        # Train
        print(f"Training {self.model_name} on {len(train_df):,} rows")
        self.train(x_train, y_train)

        # Evaluate
        val_preds = self.model.predict(x_val)
        test_preds = self.model.predict(x_test)
        val_metrics = self.evaluate(y_val, val_preds)
        test_metrics = self.evaluate(y_test, test_preds)

        print(
            f"Val:  MAE={val_metrics['mae']:.2f}  "
            f"RMSE={val_metrics['rmse']:.2f}  "
            f"MAPE={val_metrics['mape']:.2f}%  "
            f"WAPE={val_metrics['wape']:.2f}%"
        )
        print(
            f"Test: MAE={test_metrics['mae']:.2f}  "
            f"RMSE={test_metrics['rmse']:.2f}  "
            f"MAPE={test_metrics['mape']:.2f}%  "
            f"WAPE={test_metrics['wape']:.2f}%"
        )

        # MLflow logging
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            mlflow.log_params(self.model_params)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("features", self.features)
            mlflow.log_param("n_train", len(train_df))
            mlflow.log_param("n_val", len(val_df))
            mlflow.log_param("n_test", len(test_df))

            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value)
            for key, value in test_metrics.items():
                mlflow.log_metric(f"test_{key}", value)

            mlflow.lightgbm.log_model(self.model, artifact_path="model")

        # Save artifact
        model_path = self.save_model()
        print(f"Model saved → {model_path}")

        return {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_path": model_path,
        }
