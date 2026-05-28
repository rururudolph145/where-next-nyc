"""Last-week-same-hour baseline predictor for NYC Taxi demand forecasting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.config import load_config


class LastWeekSameHourBaseline:
    """Predict each (zone, bucket) as the count from one week earlier.

    The join uses wall-clock subtraction (``hour - pd.Timedelta(days=7)``),
    which works at any bucket width without bucket-integer arithmetic.
    """

    def __init__(self, target_col: str, bucket: str) -> None:
        """Initialise the baseline.

        Args:
            target_col: Name of the target column (from ``config["train"]["target"]``).
            bucket: Bucket width string (from ``config["feature"]["bucket"]``).
                Recorded for logging only; prediction uses wall-clock subtraction.
        """
        self.target_col = target_col
        self.bucket = bucket

    @classmethod
    def from_config(
        cls, config: dict[str, Any] | None = None
    ) -> LastWeekSameHourBaseline:
        """Construct from a configuration dictionary.

        Args:
            config: Parsed configuration. If ``None``, loads from ``config.yml``.

        Returns:
            Configured :class:`LastWeekSameHourBaseline` instance.
        """
        cfg = config or load_config()
        return cls(
            target_col=cfg["train"]["target"],
            bucket=cfg["feature"]["bucket"],
        )

    def predict(self, test_df: pd.DataFrame, gold_df: pd.DataFrame) -> pd.DataFrame:
        """Generate baseline predictions aligned to the test set.

        For each ``(pulocationid, hour)`` in *test_df*, look up the actual
        count from one week earlier in *gold_df*.  Rows for which the
        week-ago timestamp is absent in *gold_df* are dropped (i.e. first-week
        test rows are excluded).

        Args:
            test_df: Test-set slice of the Gold DataFrame. Must contain
                ``pulocationid``, ``hour``, and ``self.target_col``.
            gold_df: Full Gold DataFrame used as the history source.

        Returns:
            DataFrame with columns:
            ``pulocationid``, ``hour``, ``y_true``, ``y_pred_baseline``.
            Rows are restricted to those where a week-ago value exists.
        """
        # Build a lookup: (pulocationid, hour) → target_col value from gold_df
        gold_lookup = gold_df.set_index(["pulocationid", "hour"])[self.target_col]

        week_ago = pd.Timedelta(days=7)

        puloc = test_df["pulocationid"].to_numpy()
        hour = test_df["hour"].values  # numpy datetime64 array

        records = []
        for loc, h, y_true in zip(
            puloc,
            pd.to_datetime(hour),
            test_df[self.target_col].to_numpy(),
        ):
            key = (loc, h - week_ago)
            if key in gold_lookup.index:
                records.append(
                    {
                        "pulocationid": int(loc),
                        "hour": h,
                        "y_true": float(y_true),
                        "y_pred_baseline": float(gold_lookup[key]),
                    }
                )

        return pd.DataFrame(
            records,
            columns=["pulocationid", "hour", "y_true", "y_pred_baseline"],
        )
