"""Data ingestion: consolidate raw daily parquets into a single Bronze file."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import AppConfig, load_config

# Default thread-pool size for parallel parquet reads. Parquet reads
# release the GIL on the I/O syscall, so threads are sufficient and we
# avoid the pickling cost of a process pool.
_DEFAULT_MAX_WORKERS = 8


class Ingestion:
    """Reads daily parquet files from the raw layer and consolidates into Bronze.

    The raw layer is expected to contain daily parquet files organised as
    ``<raw_path>/YYYY_MM/YYYY_MM_DD.parquet``.  This class scans the
    configured date range, concatenates all matching files, and writes
    a single Bronze parquet.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialise Ingestion with configuration.

        Args:
            config: Parsed AppConfig. If ``None``, loads from ``config.yml``
                via :func:`app.core.config.load_config`.
        """
        loaded = load_config()
        self.config: AppConfig = config if config is not None else loaded  # type: ignore[assignment]
        self.raw_path: str = self.config.data.raw_path
        self.bronze_path: str = self.config.data.bronze_path
        self.start_ym: str = self.config.ingest.start_year_month
        self.end_ym: str = self.config.ingest.end_year_month
        self.max_workers: int = (
            self.config.ingest.max_workers
        )  # source: ingest.py:43 fallback
        self._assert_slice_matches_bucket()

    def _assert_slice_matches_bucket(self) -> None:
        """Guard that ``ingest.slice_duration`` == ``feature.bucket``.

        A mismatch would make the DAG advance the watermark by a different
        width than the feature aggregation bucket, silently dropping or
        double-counting demand. Both fields are required in AppConfig so the
        None-escape-hatch is removed — the guard always asserts.

        Behavior change (intentional): previously the guard was a no-op when
        either key was absent; now it always runs. Any config missing either
        field would have failed AppConfig.model_validate before reaching here.
        """
        slice_duration = self.config.ingest.slice_duration
        bucket = self.config.feature.bucket
        assert pd.Timedelta(slice_duration) == pd.Timedelta(bucket), (
            f"ingest.slice_duration ({slice_duration!r}) must equal "
            f"feature.bucket ({bucket!r}); the watermark advance width must "
            "match the feature aggregation width."
        )

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

        Reads each ``YYYY_MM/*.parquet`` in parallel via a
        ``ThreadPoolExecutor``. Parquet I/O releases the GIL, so threads
        let independent file reads overlap. Order-independence is
        preserved by a stable sort on ``tpep_pickup_datetime`` after
        ``pd.concat`` — this is the canonical contract enforced by
        ``tests/test_dataset.py::test_load_raw_parallel_equivalence``.

        Returns:
            Concatenated, time-sorted DataFrame of all raw daily files.

        Raises:
            FileNotFoundError: If no parquet files are found in the range.
        """
        tags = self._iter_year_months(self.start_ym, self.end_ym)
        raw_root = Path(self.raw_path)

        schema: dict[str, str] = self.config.ingest.bronze_schema
        canonical: dict[str, str] = {name.lower(): name for name in schema}

        def _read_one(pq_file: Path) -> pd.DataFrame:
            chunk = pd.read_parquet(pq_file)
            rename_map = {
                c: canonical[c.lower()]
                for c in chunk.columns
                if c.lower() in canonical and c != canonical[c.lower()]
            }
            return chunk.rename(columns=rename_map) if rename_map else chunk

        paths: list[Path] = []
        for tag in tags:
            month_dir = raw_root / tag
            if not month_dir.is_dir():
                print(f"  Warning: {month_dir} not found, skipping.")
                continue
            parquets = sorted(month_dir.glob("*.parquet"))
            if not parquets:
                print(f"  Warning: no parquet files in {month_dir}, skipping.")
                continue
            paths.extend(parquets)
            print(f"  {tag}: queued {len(parquets)} file(s)")

        if not paths:
            raise FileNotFoundError(
                f"No raw parquet files found under {raw_root} "
                f"for range {self.start_ym} → {self.end_ym}."
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            frames = list(executor.map(_read_one, paths))

        df = pd.concat(frames, ignore_index=True)
        # Restore canonical temporal order. Stable mergesort keeps rows
        # from the same timestamp in their submission order, so the
        # output is deterministic regardless of how parallel reads
        # interleaved.
        if "tpep_pickup_datetime" in df.columns:
            df = df.sort_values("tpep_pickup_datetime", kind="mergesort").reset_index(
                drop=True
            )
        return df

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast DataFrame columns to the types defined in config bronze_schema.

        Ensures consistent dtypes after concatenating raw files that may
        store the same column with different types across months.

        Args:
            df: Concatenated raw DataFrame.

        Returns:
            DataFrame with columns cast to the configured bronze schema.
        """
        schema: dict[str, str] = self.config.ingest.bronze_schema
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

    @staticmethod
    def _filter_window(
        df: pd.DataFrame,
        window_start: datetime | None,
        window_end: datetime | None,
    ) -> pd.DataFrame:
        """Filter *df* to ``[window_start, window_end)`` on pickup time.

        Left-inclusive / right-exclusive (``>= start`` and ``< end``). Bounds
        are compared NAIVE (the data's ``tpep_pickup_datetime`` is tz-naive
        NYC wall-clock); any tz-aware bound is stripped to naive defensively.
        ``None`` for both bounds → no filtering (backward compatible
        full-range ingest); a single ``None`` bound is treated as unbounded
        on that side.
        """
        if window_start is None and window_end is None:
            return df
        if "tpep_pickup_datetime" not in df.columns:
            return df

        pickup = df["tpep_pickup_datetime"]
        mask = pd.Series(True, index=df.index)
        if window_start is not None:
            mask &= pickup >= Ingestion._naive(window_start)
        if window_end is not None:
            mask &= pickup < Ingestion._naive(window_end)
        return df[mask]

    @staticmethod
    def _naive(dt: datetime) -> datetime:
        """Strip tzinfo so a bound is comparable to the naive data timeline."""
        return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

    def run(
        self,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> str:
        """Orchestrate the full raw → bronze consolidation.

        Args:
            window_start: Optional inclusive lower bound (NAIVE) on
                ``tpep_pickup_datetime``. When both bounds are ``None`` the
                full configured range is ingested (backward compatible).
            window_end: Optional exclusive upper bound (NAIVE).

        Returns:
            Path to the written bronze parquet file.
        """
        print(f"Ingestion: consolidating raw files ({self.start_ym} → {self.end_ym})")
        df = self.load_raw()
        df = self._filter_window(df, window_start, window_end)
        if window_start is not None or window_end is not None:
            print(f"  Window [{window_start}, {window_end}) → {len(df):,} rows")
        print(f"  Total raw rows: {len(df):,}")
        return self.save_bronze(df)
