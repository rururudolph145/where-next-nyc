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
from app.core.metrics import compute_recall_at_k, compute_wape_lift, evaluate
from app.training.baseline import LastWeekSameHourBaseline
from app.training.registry_gate import apply_gate

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
        self.tracking_uri: str = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
        self.registry_model_name: str = mlflow_cfg.get(
            "model_name", "ny_taxi_demand_model"
        )

        gate_cfg: dict[str, Any] = self.config.get("train", {}).get("gate", {})
        self.min_wape_lift_pct: float = float(gate_cfg.get("min_wape_lift_pct", 2.0))

        mlflow.set_tracking_uri(self.tracking_uri)

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

        Delegates to :func:`app.core.metrics.evaluate`.

        Args:
            y_true: True target values.
            y_pred: Predicted values.

        Returns:
            Dictionary with keys ``mae``, ``rmse``, ``mape``, ``wape``.
        """
        return evaluate(y_true, y_pred)

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
        4. Evaluate on val and test sets (full, unfiltered).
        5. Run last-week-same-hour baseline on the NaN-filtered test subset.
        6. Log params, metrics (val, test, baseline, lift, recall) to MLflow.
        7. Register model in MLflow Model Registry; run champion-vs-challenger gate.
        8. Save model artifact to disk only when the gate promotes the challenger.

        Args:
            data: Optional Gold layer DataFrame. If ``None``, loads from
                the configured gold path.

        Returns:
            Dictionary with ``val_metrics``, ``test_metrics``,
            ``baseline_metrics``, ``model_aligned_metrics``,
            ``wape_lift_pct``, ``recall_at_3``, and ``model_path``.
            ``baseline_metrics``, ``model_aligned_metrics``,
            ``wape_lift_pct``, and ``recall_at_3`` are ``None`` when the
            test split has no week-ago overlap (empty ``aligned_df``).
            ``model_path`` is ``None`` when the gate rejects the challenger.
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

        # Evaluate (full unfiltered sets — kept for backward compatibility)
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

        # Baseline evaluation on NaN-filtered test subset
        feature_cfg: dict[str, Any] = self.config.get("feature", {})
        bucket = str(feature_cfg.get("bucket", "h"))
        baseline = LastWeekSameHourBaseline(target_col=self.target, bucket=bucket)
        aligned_df = baseline.predict(test_df, data)

        # Filter model predictions to the same (pulocationid, hour) index
        common_idx = pd.MultiIndex.from_arrays(
            [aligned_df["pulocationid"], aligned_df["hour"]]
        )
        test_mi = pd.MultiIndex.from_arrays([test_df["pulocationid"], test_df["hour"]])
        mask = test_mi.isin(common_idx)
        test_df_aligned = test_df.loc[mask]
        test_preds_aligned = test_preds[mask]

        # Guard: if aligned_df is empty (no week-ago overlap) skip all
        # baseline/lift/recall computation — logging NaN metrics to MLflow
        # would produce meaningless run records.
        baseline_metrics: dict[str, float] | None
        model_aligned_metrics: dict[str, float] | None
        wape_lift_pct: float | None
        recall_at_3: float | None

        if len(aligned_df) == 0:
            print(
                "No week-ago data overlap for test split; "
                "skipping baseline + lift + recall@K metrics."
            )
            baseline_metrics = None
            model_aligned_metrics = None
            wape_lift_pct = None
            recall_at_3 = None
        else:
            model_aligned_metrics = self.evaluate(
                test_df_aligned[self.target], test_preds_aligned
            )
            baseline_metrics = self.evaluate(
                aligned_df["y_true"],
                aligned_df["y_pred_baseline"].to_numpy(),
            )

            wape_lift_pct = compute_wape_lift(
                baseline_metrics["wape"], model_aligned_metrics["wape"]
            )

            # Recall@3
            predicted_df = pd.DataFrame(
                {
                    "pulocationid": test_df_aligned["pulocationid"].values,
                    "hour": test_df_aligned["hour"].values,
                    "value": test_preds_aligned,
                }
            )
            actual_df = pd.DataFrame(
                {
                    "pulocationid": aligned_df["pulocationid"].values,
                    "hour": aligned_df["hour"].values,
                    "value": aligned_df["y_true"].values,
                }
            )
            recall_at_3 = compute_recall_at_k(predicted_df, actual_df, k=3)

            print(
                f"Baseline WAPE={baseline_metrics['wape']:.2f}%  "
                f"Model WAPE={model_aligned_metrics['wape']:.2f}%  "
                f"Lift={wape_lift_pct:.2f}%  Recall@3={recall_at_3:.4f}"
            )

        # MLflow logging
        mlflow.set_experiment(self.experiment_name)

        model_path: str | None = None

        with mlflow.start_run() as active_run:
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

            # Only log baseline/lift/recall metrics when they are valid numbers.
            if baseline_metrics is not None and model_aligned_metrics is not None:
                mlflow.log_metric("baseline_test_mae", baseline_metrics["mae"])
                mlflow.log_metric("baseline_test_wape", baseline_metrics["wape"])
                mlflow.log_metric("model_test_mae", model_aligned_metrics["mae"])
                mlflow.log_metric("model_test_wape", model_aligned_metrics["wape"])
            if wape_lift_pct is not None:
                mlflow.log_metric("wape_lift_pct", wape_lift_pct)
            if recall_at_3 is not None:
                mlflow.log_metric("recall_at_3", recall_at_3)

            mlflow.lightgbm.log_model(self.model, artifact_path="model")

            # Register model in MLflow Model Registry.
            run_id = active_run.info.run_id
            run_uri = f"runs:/{run_id}/model"
            registered = mlflow.register_model(run_uri, self.registry_model_name)
            new_version: str = registered.version

            # Tag the new version with challenger WAPE for future gate reads.
            # Skip when wape_lift_pct is None (no baseline overlap) — the tag
            # would be meaningless and would look like a missing champion WAPE
            # to a future run.
            gate_result: str
            rejection_reason: str | None
            client = mlflow.tracking.MlflowClient()
            if wape_lift_pct is not None and model_aligned_metrics is not None:
                mlflow.set_tag("model_test_wape", str(model_aligned_metrics["wape"]))
                client.set_model_version_tag(
                    self.registry_model_name,
                    new_version,
                    "model_test_wape",
                    str(model_aligned_metrics["wape"]),
                )
                challenger_wape = model_aligned_metrics["wape"]
                gate_result, rejection_reason = apply_gate(
                    client,
                    self.registry_model_name,
                    new_version,
                    challenger_wape,
                    self.min_wape_lift_pct,
                )
            else:
                # No baseline — check whether an incumbent exists.
                production_versions = client.get_latest_versions(
                    self.registry_model_name, stages=["Production"]
                )
                if production_versions:
                    gate_result = "rejected"
                    rejection_reason = "no_baseline_for_comparison"
                else:
                    # First-ever run without baseline — auto-promote.
                    client.transition_model_version_stage(
                        name=self.registry_model_name,
                        version=new_version,
                        stage="Production",
                    )
                    gate_result = "promoted"
                    rejection_reason = None

            mlflow.set_tag("gate_result", gate_result)
            if rejection_reason is not None:
                mlflow.set_tag("gate_rejection_reason", rejection_reason)

            if gate_result == "promoted":
                # Write wape_lift_pct tag only when the value is finite and
                # computable — skip None (no baseline) and inf (zero-WAPE baseline).
                if wape_lift_pct is not None and wape_lift_pct != float("inf"):
                    client.set_model_version_tag(
                        self.registry_model_name,
                        new_version,
                        "wape_lift_pct",
                        str(wape_lift_pct),
                    )
                model_path = self.save_model()
                print(f"Model saved → {model_path}")
            else:
                print(
                    f"Gate rejected challenger "
                    f"(reason: {rejection_reason}); disk model unchanged."
                )

        return {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "baseline_metrics": baseline_metrics,
            "model_aligned_metrics": model_aligned_metrics,
            "wape_lift_pct": wape_lift_pct,
            "recall_at_3": recall_at_3,
            "model_path": model_path,
        }
