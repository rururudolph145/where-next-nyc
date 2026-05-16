"""Data ingestion: consolidate raw daily parquets into a single Bronze file."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.shared.config import load_config


class Ingestion:
    """Reads daily parquet files from the raw layer and consolidates into Bronze.

    The raw layer is expected to contain daily parquet files organised as
    ``<raw_path>/YYYY_MM/YYYY_MM_DD.parquet``.  This class scans the
    configured date range, concatenates all matching files, and writes
    a single Bronze parquet.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise Ingestion with configuration.

        Args:
            config: Parsed configuration dictionary.  If ``None``, loads
                from ``config.yml`` via :func:`src.shared.config.load_config`.
        """
        self.config = config or load_config()
        self.raw_path: str = self.config["data"]["raw_path"]
        self.bronze_path: str = self.config["data"]["bronze_path"]
        self.start_ym: str = self.config["ingest"]["start_year_month"]
        self.end_ym: str = self.config["ingest"]["end_year_month"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_year_months(start: str, end: str) -> list[str]:
        """Return ``YYYY_MM`` folder names from *start* to *end* inclusive.

        Args:
            start: Year-month string ``YYYY-MM``.
            end: Year-month string ``YYYY-MM``.

        Returns:
            List of folder-name strings like ``["2021_01", "2021_02"]``.
        """
        s_year, s_month = (int(x) for x in start.split("-"))
        e_year, e_month = (int(x) for x in end.split("-"))

        tags: list[str] = []
        year, month = s_year, s_month
        while (year, month) <= (e_year, e_month):
            tags.append(f"{year}_{month:02d}")
            month += 1
            if month > 12:
                month = 1
                year += 1
        return tags

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def load_raw(self) -> pd.DataFrame:
        """Scan the raw folder for daily parquets and concatenate them.

        Iterates through each ``YYYY_MM`` subfolder within the configured
        date range and reads every ``.parquet`` file found.

        Returns:
            Concatenated DataFrame of all raw daily files.

        Raises:
            FileNotFoundError: If no parquet files are found in the range.
        """
        tags = self._iter_year_months(self.start_ym, self.end_ym)
        raw_root = Path(self.raw_path)

        schema: dict[str, str] = self.config["ingest"]["bronze_schema"]
        canonical: dict[str, str] = {name.lower(): name for name in schema}

        frames: list[pd.DataFrame] = []
        for tag in tags:
            month_dir = raw_root / tag
            if not month_dir.is_dir():
                print(f"  Warning: {month_dir} not found, skipping.")
                continue
            parquets = sorted(month_dir.glob("*.parquet"))
            if not parquets:
                print(f"  Warning: no parquet files in {month_dir}, skipping.")
                continue
            for pq_file in parquets:
                chunk = pd.read_parquet(pq_file)
                rename_map = {
                    c: canonical[c.lower()]
                    for c in chunk.columns
                    if c.lower() in canonical and c != canonical[c.lower()]
                }
                if rename_map:
                    chunk = chunk.rename(columns=rename_map)
                frames.append(chunk)
            print(f"  {tag}: read {len(parquets)} file(s)")

        if not frames:
            raise FileNotFoundError(
                f"No raw parquet files found under {raw_root} "
                f"for range {self.start_ym} → {self.end_ym}."
            )

        return pd.concat(frames, ignore_index=True)

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast DataFrame columns to the types defined in config bronze_schema.

        Ensures consistent dtypes after concatenating raw files that may
        store the same column with different types across months.

        Args:
            df: Concatenated raw DataFrame.

        Returns:
            DataFrame with columns cast to the configured bronze schema.
        """
        schema: dict[str, str] = self.config["ingest"]["bronze_schema"]
        for col, dtype in schema.items():
            if col not in df.columns:
                continue
            if dtype.startswith("datetime64"):
                df[col] = pd.to_datetime(df[col])
            elif dtype == "str":
                df[col] = df[col].astype(str)
            else:
                if df[col].dtype == object:
                    df[col] = df[col].map(
                        lambda v: v.decode() if isinstance(v, bytes) else v
                    )
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                    np.dtype(dtype)
                )
        return df

    def save_bronze(self, df: pd.DataFrame) -> str:
        """Write a single consolidated parquet to the Bronze layer.

        Enforces the configured bronze schema before writing to ensure
        consistent column types across different source files.

        Args:
            df: DataFrame to persist.

        Returns:
            The bronze file path that was written.
        """
        os.makedirs(os.path.dirname(self.bronze_path), exist_ok=True)
        df = self._enforce_schema(df)
        df.to_parquet(self.bronze_path, index=False)
        print(f"Saved → {self.bronze_path}  ({len(df):,} rows)")
        return self.bronze_path

    def run(self) -> str:
        """Orchestrate the full raw → bronze consolidation.

        Returns:
            Path to the written bronze parquet file.
        """
        print(f"Ingestion: consolidating raw files ({self.start_ym} → {self.end_ym})")
        df = self.load_raw()
        print(f"  Total raw rows: {len(df):,}")
        return self.save_bronze(df)

    # ------------------------------------------------------------------
    # Legacy helper kept for downstream compatibility
    # ------------------------------------------------------------------

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the train and test parquet files specified in config.

        Returns:
            A ``(train, test)`` tuple of DataFrames.
        """
        train_path: str = self.config["data"]["train_path"]
        test_path: str = self.config["data"]["test_path"]
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        return train, test
