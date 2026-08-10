"""Batch inference for NYC Taxi demand forecasting.

Loads a trained model, generates predictions on Gold layer data,
builds a predictions table, evaluates metrics, and saves results.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import joblib  # type: ignore[import-untyped]
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

from app.core.config import load_config
from app.core.metrics import evaluate as _evaluate

logger = logging.getLogger(__name__)


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

        mlflow_cfg: dict[str, Any] = self.config.get("mlflow", {})
        self.tracking_uri: str = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
        self.registry_model_name: str = mlflow_cfg.get(
            "model_name", "ny_taxi_demand_model"
        )

        self.model: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> Any:
        """Load the trained model — registry primary, disk fallback.

        Tries the @champion alias in the MLflow registry first.  Falls back to
        ``<store_path>/model.pkl`` only when the registry raises
        ``MlflowException`` (alias absent, registry unreachable, etc.).
        A bare ``Exception`` from FS or attribute errors is NOT caught here so
        genuine bugs surface immediately.

        Returns:
            The deserialized model object.

        Raises:
            FileNotFoundError: If the registry is unavailable AND the model
                file does not exist at the fallback disk path.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        champion_uri = f"models:/{self.registry_model_name}@champion"
        try:
            self.model = mlflow.sklearn.load_model(champion_uri)
            return self.model
        except MlflowException:
            logger.warning(
                "Registry @champion unavailable; falling back to disk: %s",
                self.store_path,
            )

        model_path = os.path.join(self.store_path, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at registry ({champion_uri}) or disk ({model_path})"
            )
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

        The target column is optional.  When absent (inference path),
        ``actual_trip_count`` is written as ``NaN`` so the deferred
        monitoring join (``actuals_join``) can fill it later.

        Args:
            df: Gold layer DataFrame with ``pulocationid`` and ``hour``.
                The target column (``trip_count``) is optional.
            preds: Predicted values from :meth:`predict`.

        Returns:
            DataFrame with columns ``pulocationid``, ``hour``,
            ``predicted_trip_count``, ``actual_trip_count``
            (NaN when target absent).
        """
        actual = (
            df[self.target].to_numpy()
            if self.target in df.columns
            else np.full(len(df), np.nan)
        )
        return pd.DataFrame(
            {
                "pulocationid": df["pulocationid"].values,
                "hour": df["hour"].values,
                "predicted_trip_count": preds,
                "actual_trip_count": actual,
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

    def _upsert_day_partition(self, history_dir: str, new_rows: pd.DataFrame) -> None:
        """Upsert new_rows into the flat history/dt=<date>.parquet for each date.

        Uses concat([existing, new_rows]) + drop_duplicates(keep="last") so a
        re-scored (pulocationid, hour) bucket keeps the freshest value.
        """
        for date_val, group in new_rows.groupby(new_rows["hour"].dt.date):
            day_path = os.path.join(history_dir, f"dt={date_val!s}.parquet")
            if os.path.exists(day_path):
                existing = pd.read_parquet(day_path)
                group = (
                    pd.concat([existing, group])
                    .drop_duplicates(subset=["pulocationid", "hour"], keep="last")
                    .reset_index(drop=True)
                )
            group.to_parquet(day_path, index=False)

    def save_predictions(self, predictions_df: pd.DataFrame) -> str:
        """Save the predictions table to a parquet file.

        Writes the hot-slot first, then upserts each touched calendar date into
        ``history/dt=<YYYY-MM-DD>.parquet``.  A history-write failure is logged
        and swallowed so the hot-slot is never affected.

        Args:
            predictions_df: DataFrame produced by :meth:`build_predictions_table`.

        Returns:
            Path to the saved parquet file.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        predictions_df.to_parquet(self.output_path, index=False)
        try:
            history_dir = os.path.join(os.path.dirname(self.output_path), "history")
            os.makedirs(history_dir, exist_ok=True)
            self._upsert_day_partition(history_dir, predictions_df)
        except Exception:
            logger.warning("History write failed; hot-slot unaffected.", exc_info=True)
        return self.output_path

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame | None = None,
        since: Any | None = None,
    ) -> dict[str, Any]:
        """Execute the full inference pipeline.

        1. Load model.
        2. Load Gold data (or use provided DataFrame).
        3. Optionally filter to rows with ``hour >= since``.
        4. If the filtered frame is empty, return early without touching the hot-slot.
        5. Generate predictions, build table, evaluate metrics, save.

        Args:
            data: Optional Gold layer DataFrame. If ``None``, loads from
                the configured gold path.
            since: Optional lower-bound timestamp (inclusive). When supplied,
                only rows with ``hour >= since`` are scored. Pass the
                ``window_start`` watermark value to restrict work to the new
                DAG slice. ``None`` preserves the full-gold behaviour.

        Returns:
            Dictionary with ``metrics``, ``output_path``, and
            ``n_predictions``.  ``metrics`` is ``{}`` when the target column
            is absent (inference path) — evaluation is a monitoring concern
            and is handled by deferred ``actuals_join``.
        """
        self.load_model()

        if data is None:
            data = self.load_gold()

        if since is not None:
            data = data[data["hour"] >= since]
            if data.empty:
                logger.info(
                    "run: since-filter emptied the frame — skipping predict/save"
                )
                return {
                    "metrics": {},
                    "output_path": self.output_path,
                    "n_predictions": 0,
                }

        preds = self.predict(data)
        predictions_df = self.build_predictions_table(data, preds)

        # Evaluate only when the target column is present and has scorable rows.
        # Live DAG ticks arrive with inference gold (no target) — metrics stay {}
        # and scoring is deferred to the monitoring actuals_join path.
        metrics: dict[str, float] = {}
        if self.target in data.columns:
            scorable = data[self.target].notna()
            if scorable.any():
                metrics = self.evaluate(
                    data.loc[scorable, self.target],
                    preds[scorable.to_numpy()],
                )
        if metrics:
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
