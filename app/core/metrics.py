"""Canonical evaluation metrics for NYC Taxi demand forecasting.

All evaluation logic is centralised here so that Trainer, Predictor, and
HourlySimulator share a single implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:  # type: ignore[type-arg]
    """Compute evaluation metrics.

    MAPE is computed only on non-zero actuals to avoid divide-by-zero.
    WAPE = sum|actual - pred| / sum|actual| × 100; returns 0.0 when all
    actuals are zero.

    Args:
        y_true: True target values.
        y_pred: Predicted values array.

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
    wape = float(np.sum(np.abs(actual - y_pred)) / total * 100) if total > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "mape": mape, "wape": wape}


def compute_recall_at_k(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    k: int = 3,
) -> float:
    """Per-bucket recall at k.

    For each unique ``hour`` in *predicted*, sort zones by ``value`` descending
    (ties broken by ``pulocationid`` ascending) and take the top-k zone IDs.
    Recall for that bucket = |top_k_pred ∩ top_k_actual| / k.

    Returns the unweighted mean recall across all bucket-times present in
    *predicted*. The denominator stays k even when a bucket has fewer than k
    zones.

    Args:
        predicted: DataFrame with columns ``pulocationid`` (int), ``hour``
            (Timestamp), ``value`` (float) — model predictions.
        actual: Same column schema — ground-truth counts.
        k: Number of top zones to compare.

    Returns:
        Mean recall at k (float in [0.0, 1.0]).
    """
    # Build lookup: hour → set of top-k actual zone IDs
    actual_top: dict[pd.Timestamp, set[int]] = {}
    for hour, grp in actual.groupby("hour", sort=False):
        sorted_grp = grp.sort_values(["value", "pulocationid"], ascending=[False, True])
        top_ids: set[int] = set(sorted_grp["pulocationid"].iloc[:k].tolist())
        actual_top[hour] = top_ids  # type: ignore[index]

    recalls: list[float] = []
    for hour, grp in predicted.groupby("hour", sort=False):
        sorted_grp = grp.sort_values(["value", "pulocationid"], ascending=[False, True])
        pred_ids: set[int] = set(sorted_grp["pulocationid"].iloc[:k].tolist())
        true_ids = actual_top.get(hour, set())  # type: ignore[arg-type]
        recalls.append(len(pred_ids & true_ids) / k)

    return float(np.mean(recalls)) if recalls else 0.0


def compute_wape_lift(baseline_wape: float, model_wape: float) -> float:
    """Percentage WAPE improvement over the baseline.

    lift = (baseline_wape - model_wape) / baseline_wape × 100

    A positive lift means the model beats the baseline; negative means it
    is worse.

    Args:
        baseline_wape: WAPE of the last-week-same-hour baseline.
        model_wape: WAPE of the trained model.

    Returns:
        Lift percentage. Returns ``float('inf')`` when *baseline_wape* is 0.
    """
    if baseline_wape == 0.0:
        return float("inf")
    return (baseline_wape - model_wape) / baseline_wape * 100
