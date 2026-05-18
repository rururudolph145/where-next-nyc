"""Hourly inference simulation for NYC Taxi demand forecasting.

Simulates real-time inference over an out-of-sample period (e.g. March
2021) by stepping through each hour and generating predictions for all
zones.  Lag features are pre-computed in the Gold layer, so the
simulation simply filters and predicts hour-by-hour.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from app.core.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


class HourlySimulator:
    """Simulate hourly demand prediction over an out-of-sample period.

    The Gold layer must already contain features and actuals for both the
    warm-up period (training months) and the simulation period.

    Attributes:
        config: Parsed configuration dictionary.
        features: List of feature column names used by the model.
        target: Name of the target column (``trip_count``).
        gold_path: Path to the Gold layer directory.
        model_path: Path to the serialized model file.
        output_path: Path for saving simulation predictions.
        start_hour: First hour of the simulation window.
        end_hour: Last hour of the simulation window.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise the simulator from configuration.

        Args:
            config: Parsed configuration dictionary. If ``None``, loads
                from ``config.yml``.
        """
        self.config: dict[str, Any] = config or load_config()

        train_cfg: dict[str, Any] = self.config["train"]
        self.features: list[str] = train_cfg["features"]
        self.target: str = train_cfg["target"]

        self.gold_path: str = self.config["data"]["gold_path"]

        model_cfg: dict[str, Any] = self.config["model"]
        self.model_path: str = os.path.join(model_cfg["store_path"], "model.pkl")

        inference_cfg: dict[str, Any] = self.config.get("inference", {})
        self.output_path: str = inference_cfg.get(
            "output_path", "data/predictions/demand_predictions.parquet"
        )

        self.heatmap_path: str = os.path.join(self.gold_path, "hourly_heatmap.parquet")

        sim_cfg: dict[str, Any] = inference_cfg.get("simulation", {})
        self.start_hour: pd.Timestamp = pd.Timestamp(
            sim_cfg.get("start_hour", "2021-03-01 00:00:00")
        )
        self.end_hour: pd.Timestamp = pd.Timestamp(
            sim_cfg.get("end_hour", "2021-03-31 23:00:00")
        )

    # ------------------------------------------------------------------
    # Data & model loading
    # ------------------------------------------------------------------

    def load_model(self) -> Any:
        """Load the trained model from disk.

        Returns:
            The deserialized model object.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        model: Any = joblib.load(self.model_path)
        return model

    def load_gold(self) -> pd.DataFrame:
        """Load the Gold layer parquet.

        Returns:
            Gold layer DataFrame with features, target, and metadata.
        """
        gold_file = os.path.join(self.gold_path, "hourly_demand_featured.parquet")
        return pd.read_parquet(gold_file)

    def get_inference_window(self, gold_df: pd.DataFrame) -> pd.DataFrame:
        """Filter the Gold DataFrame to the simulation window.

        Args:
            gold_df: Full Gold layer DataFrame.

        Returns:
            Subset of rows whose ``hour`` falls within
            ``[start_hour, end_hour]``.
        """
        mask = (gold_df["hour"] >= self.start_hour) & (gold_df["hour"] <= self.end_hour)
        return gold_df.loc[mask].copy()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(
        actuals: np.ndarray[Any, np.dtype[np.floating[Any]]],
        predictions: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ) -> dict[str, float]:
        """Compute aggregate error metrics.

        Args:
            actuals: Ground truth values.
            predictions: Predicted values.

        Returns:
            Dictionary with ``mae``, ``rmse``, ``mape``, ``wape``.
        """
        mae = float(np.mean(np.abs(actuals - predictions)))
        rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))

        mask = actuals != 0
        if bool(mask.any()):
            mape = float(
                np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask]))
                * 100
            )
        else:
            mape = 0.0

        total = float(np.sum(np.abs(actuals)))
        wape = (
            float(np.sum(np.abs(actuals - predictions)) / total * 100)
            if total > 0
            else 0.0
        )

        return {"mae": mae, "rmse": rmse, "mape": mape, "wape": wape}

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the hourly simulation.

        Steps through each hour in the simulation window, predicts
        demand for all zones, and computes evaluation metrics.

        Returns:
            Dictionary with keys:
            - ``predictions_df``: DataFrame of per-zone, per-hour predictions.
            - ``metrics``: Aggregate metrics dictionary.
            - ``per_zone_metrics``: Per-zone metrics DataFrame.
            - ``output_path``: Path where predictions were saved.
            - ``heatmap_path``: Path to the Gold heatmap parquet.
            - ``n_hours``: Number of simulation hours.
            - ``n_predictions``: Total number of prediction rows.
        """
        model = self.load_model()
        gold_df = self.load_gold()

        logging.info(
            "Gold layer loaded: %s rows, hour range %s → %s",
            f"{len(gold_df):,}",
            gold_df["hour"].min(),
            gold_df["hour"].max(),
        )

        sim_df = self.get_inference_window(gold_df)
        logging.info(
            "Simulation window: %s → %s (%s rows)",
            self.start_hour,
            self.end_hour,
            f"{len(sim_df):,}",
        )

        if sim_df.empty:
            raise ValueError(
                f"No data in simulation window {self.start_hour} → {self.end_hour}"
            )

        # Sort by hour to simulate chronological stepping
        sim_df = sim_df.sort_values("hour")
        unique_hours = sim_df["hour"].unique()

        records: list[dict[str, Any]] = []

        for i, hour_ts in enumerate(unique_hours):
            hour_slice = sim_df.loc[sim_df["hour"] == hour_ts]
            x = hour_slice[self.features]
            preds: np.ndarray[Any, np.dtype[np.floating[Any]]] = model.predict(x)

            for zone_id, predicted, actual in zip(
                hour_slice["pulocationid"].values,
                preds,
                hour_slice[self.target].values,
                strict=True,
            ):
                records.append(
                    {
                        "pulocationid": zone_id,
                        "hour": hour_ts,
                        "predicted_trip_count": float(predicted),
                        "actual_trip_count": float(actual),
                    }
                )

            if (i + 1) % 100 == 0:
                logging.info("  Processed %d / %d hours", i + 1, len(unique_hours))

        logging.info(
            "Simulation complete: %d hours, %d predictions",
            len(unique_hours),
            len(records),
        )

        predictions_df = pd.DataFrame(records)

        # --- Aggregate metrics ---
        actuals_arr = predictions_df["actual_trip_count"].to_numpy(dtype=np.float64)
        preds_arr = predictions_df["predicted_trip_count"].to_numpy(dtype=np.float64)
        metrics = self.compute_metrics(actuals_arr, preds_arr)

        logging.info(
            "Aggregate metrics: MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  WAPE=%.2f%%",
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
            metrics["wape"],
        )

        # --- Per-zone metrics ---
        zone_metrics_records: list[dict[str, Any]] = []
        for zone_id, grp in predictions_df.groupby("pulocationid"):
            z_actuals = grp["actual_trip_count"].to_numpy(dtype=np.float64)
            z_preds = grp["predicted_trip_count"].to_numpy(dtype=np.float64)
            z_metrics: dict[str, Any] = {
                "pulocationid": zone_id,
                **self.compute_metrics(z_actuals, z_preds),
            }
            zone_metrics_records.append(z_metrics)

        per_zone_metrics = pd.DataFrame(zone_metrics_records)

        # --- Save predictions ---
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        predictions_df.to_parquet(self.output_path, index=False)
        logging.info(
            "Predictions saved → %s (%s rows)",
            self.output_path,
            f"{len(predictions_df):,}",
        )

        # --- Save hourly heatmap (zone × hour predictions) ---
        os.makedirs(os.path.dirname(self.heatmap_path), exist_ok=True)
        predictions_df.to_parquet(self.heatmap_path, index=False)
        logging.info("Heatmap saved → %s", self.heatmap_path)

        return {
            "predictions_df": predictions_df,
            "metrics": metrics,
            "per_zone_metrics": per_zone_metrics,
            "output_path": self.output_path,
            "heatmap_path": self.heatmap_path,
            "n_hours": len(unique_hours),
            "n_predictions": len(predictions_df),
        }
