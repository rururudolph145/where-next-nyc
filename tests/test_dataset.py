from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import mock_open, patch

from app.core.config import AppConfig
from app.ingestion.ingest import Ingestion


@pytest.fixture
def config_data() -> AppConfig:
    from tests.conftest import _FULL_CONFIG_DICT

    return AppConfig.model_validate(_FULL_CONFIG_DICT)


@pytest.fixture
def sample_data():
    train = pd.DataFrame({"hour": [8, 9], "trip_count": [100, 150]})
    test = pd.DataFrame({"hour": [10], "trip_count": [120]})
    return train, test


@pytest.mark.skip(
    reason="Ingestion.__init__ reads 'raw_path' key absent from this fixture; root cause fix deferred"
)
@patch("builtins.open", new_callable=mock_open, read_data="dummy")
@patch("yaml.safe_load")
@patch("pandas.read_parquet")
def test_load_data(
    mock_read_parquet, mock_safe_load, mock_open_file, config_data, sample_data
):
    mock_safe_load.return_value = config_data
    mock_read_parquet.side_effect = sample_data

    ingestion = Ingestion(config=config_data)
    train_data, test_data = ingestion.load_data()

    pd.testing.assert_frame_equal(train_data, sample_data[0])
    pd.testing.assert_frame_equal(test_data, sample_data[1])

    mock_read_parquet.assert_any_call("data/train.parquet")
    mock_read_parquet.assert_any_call("data/test.parquet")


# -- PBI 12: Bronze temporal-sort invariant ----------------------------------
#
# Pins the contract that Bronze data is globally sorted by
# tpep_pickup_datetime. Locks the invariant before PBI 14 introduces
# parallel reads, which would otherwise break ordering silently.


def _write_daily_parquet(path: Path, start: pd.Timestamp, n_rows: int) -> None:
    """Write a tiny daily parquet whose rows are internally time-sorted."""
    timestamps = pd.date_range(start, periods=n_rows, freq="15min")
    df = pd.DataFrame(
        {
            "VendorID": [1.0] * n_rows,
            "tpep_pickup_datetime": timestamps,
            "tpep_dropoff_datetime": timestamps + pd.Timedelta(minutes=10),
            "passenger_count": [1.0] * n_rows,
            "trip_distance": [1.5] * n_rows,
            "PULocationID": [100.0] * n_rows,
            "DOLocationID": [200.0] * n_rows,
            "fare_amount": [10.0] * n_rows,
        }
    )
    df.to_parquet(path, index=False)


@pytest.fixture
def bronze_schema() -> dict[str, str]:
    """Subset of the production bronze schema sufficient for the fixture."""
    return {
        "VendorID": "float64",
        "tpep_pickup_datetime": "datetime64[us]",
        "tpep_dropoff_datetime": "datetime64[us]",
        "passenger_count": "float64",
        "trip_distance": "float64",
        "PULocationID": "float64",
        "DOLocationID": "float64",
        "fare_amount": "float64",
    }


@pytest.fixture
def ordered_raw_tree(tmp_path: Path) -> Path:
    """Create raw/YYYY_MM/YYYY_MM_DD.parquet files in temporal order."""
    raw_root = tmp_path / "raw"
    schedule = [
        ("2021_01", "2021_01_01", pd.Timestamp("2021-01-01 00:00:00")),
        ("2021_01", "2021_01_02", pd.Timestamp("2021-01-02 00:00:00")),
        ("2021_02", "2021_02_01", pd.Timestamp("2021-02-01 00:00:00")),
        ("2021_02", "2021_02_02", pd.Timestamp("2021-02-02 00:00:00")),
    ]
    for month_tag, day_tag, start in schedule:
        month_dir = raw_root / month_tag
        month_dir.mkdir(parents=True, exist_ok=True)
        _write_daily_parquet(month_dir / f"{day_tag}.parquet", start, n_rows=4)
    return raw_root


def test_bronze_is_temporally_sorted(
    tmp_path: Path, ordered_raw_tree: Path, bronze_schema: dict[str, str]
) -> None:
    """Bronze output must be globally sorted by tpep_pickup_datetime.

    Locks the invariant against the current sequential implementation.
    PBI 14 (parallel reads) must preserve this property.
    """
    bronze_path = tmp_path / "bronze" / "yellow_tripdata" / "bronze.parquet"
    from tests.conftest import _FULL_CONFIG_DICT

    base = AppConfig.model_validate(_FULL_CONFIG_DICT)
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "raw_path": str(ordered_raw_tree),
                    "bronze_path": str(bronze_path),
                }
            ),
            "ingest": base.ingest.model_copy(
                update={
                    "start_year_month": "2021-01",
                    "end_year_month": "2021-02",
                    "bronze_schema": bronze_schema,
                }
            ),
        }
    )

    Ingestion(config=config).run()

    bronze_df = pd.read_parquet(bronze_path)
    assert bronze_df["tpep_pickup_datetime"].is_monotonic_increasing


# -- PBI 13: Parallel-equivalence regression --------------------------------
#
# Pins the contract that ingestion is order-independent: regardless of the
# order in which raw files are read (glob order, ThreadPoolExecutor
# completion order, etc.), Bronze must end up globally time-sorted and
# equal to the canonical reference. RED today (current concat does not
# sort); GREEN after PBI 14 adds the post-concat stable sort.


@pytest.fixture
def out_of_order_raw_tree(tmp_path: Path) -> Path:
    """Create raw files whose alphabetical glob order ≠ temporal order.

    Same month directory, two daily parquets. The file that sorts first
    by name contains the *later* timestamps and vice versa, so the
    sequential ``for pq_file in sorted(parquets)`` loop emits rows
    out-of-order. This mimics the hazard PBI 14's parallel reads will
    introduce (ThreadPoolExecutor completion order is non-deterministic).
    """
    raw_root = tmp_path / "raw"
    month_dir = raw_root / "2021_01"
    month_dir.mkdir(parents=True, exist_ok=True)

    # Alphabetically first filename → later timestamps
    _write_daily_parquet(
        month_dir / "2021_01_01.parquet",
        start=pd.Timestamp("2021-01-15 00:00:00"),
        n_rows=4,
    )
    # Alphabetically second filename → earlier timestamps
    _write_daily_parquet(
        month_dir / "2021_01_02.parquet",
        start=pd.Timestamp("2021-01-05 00:00:00"),
        n_rows=4,
    )
    return raw_root


def test_load_raw_parallel_equivalence(
    tmp_path: Path, out_of_order_raw_tree: Path, bronze_schema: dict[str, str]
) -> None:
    """Bronze output must equal the canonical sorted reference regardless
    of file read order.

    The fixture intentionally arranges files so that glob/concat order
    yields a temporally-shuffled DataFrame — exactly what parallel reads
    can produce when ThreadPoolExecutor.map preserves submission order
    but the wall-clock cost per file varies. The assertion fails today
    because the current ``load_raw`` performs no post-concat sort.
    PBI 14 must add ``sort_values("tpep_pickup_datetime", kind="mergesort")``
    after concat to satisfy this contract.
    """
    bronze_path = tmp_path / "bronze" / "yellow_tripdata" / "bronze.parquet"
    from tests.conftest import _FULL_CONFIG_DICT

    base = AppConfig.model_validate(_FULL_CONFIG_DICT)
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "raw_path": str(out_of_order_raw_tree),
                    "bronze_path": str(bronze_path),
                }
            ),
            "ingest": base.ingest.model_copy(
                update={
                    "start_year_month": "2021-01",
                    "end_year_month": "2021-01",
                    "bronze_schema": bronze_schema,
                }
            ),
        }
    )

    Ingestion(config=config).run()
    bronze_df = pd.read_parquet(bronze_path)

    # Canonical reference: read all raw files, concat, then stable-sort
    # by pickup time. PBI 14 must produce exactly this output regardless
    # of how it parallelized the reads.
    raw_files = sorted((out_of_order_raw_tree / "2021_01").glob("*.parquet"))
    reference = (
        pd.concat([pd.read_parquet(p) for p in raw_files], ignore_index=True)
        .sort_values("tpep_pickup_datetime", kind="mergesort")
        .reset_index(drop=True)
    )
    # Align dtypes against what _enforce_schema + parquet round-trip
    # actually produce in Bronze: pd.to_datetime → datetime64[ns]
    # regardless of the config's [us] declaration (a pre-existing
    # behavior of _enforce_schema, out of scope for PBI 14).
    reference["tpep_pickup_datetime"] = pd.to_datetime(
        reference["tpep_pickup_datetime"]
    )
    reference["tpep_dropoff_datetime"] = pd.to_datetime(
        reference["tpep_dropoff_datetime"]
    )

    assert bronze_df["tpep_pickup_datetime"].is_monotonic_increasing
    pd.testing.assert_frame_equal(
        bronze_df.reset_index(drop=True),
        reference[bronze_df.columns.tolist()].reset_index(drop=True),
        check_like=False,
    )


# ---------------------------------------------------------------------------
# Steps 1–3: Ingestion.append_slice (pure, no I/O)
# ---------------------------------------------------------------------------


def _make_df(timestamps: list[datetime]) -> pd.DataFrame:
    """Build a minimal DataFrame with tpep_pickup_datetime for append_slice tests."""
    return pd.DataFrame({"tpep_pickup_datetime": pd.to_datetime(timestamps)})


class TestAppendSlice:
    """Ingestion.append_slice merges slices idempotently."""

    def test_append_slice_concats_disjoint_slices(self) -> None:
        """Result rowcount equals the sum of both slices, sorted by pickup time.

        AC: new rows appended, output sorted by tpep_pickup_datetime.
        """
        # Arrange
        cutoff = datetime(2021, 1, 19)
        existing = _make_df([datetime(2021, 1, 18, 0), datetime(2021, 1, 18, 1)])
        new_slice = _make_df([datetime(2021, 1, 19, 0), datetime(2021, 1, 19, 1)])

        # Act
        result = Ingestion.append_slice(existing, new_slice, cutoff=cutoff)

        # Assert
        assert len(result) == 4
        assert result["tpep_pickup_datetime"].is_monotonic_increasing

    def test_append_slice_is_idempotent_on_same_window(self) -> None:
        """Re-appending the same slice yields the same rowcount as a clean append.

        AC: retry-safe — no row duplication on repeated runs of the same window.
        """
        # Arrange
        cutoff = datetime(2021, 1, 19)
        existing = _make_df([datetime(2021, 1, 18, 0)])
        new_slice = _make_df([datetime(2021, 1, 19, 0)])

        # Act
        first = Ingestion.append_slice(existing, new_slice, cutoff=cutoff)
        second = Ingestion.append_slice(first, new_slice, cutoff=cutoff)

        # Assert
        assert len(second) == len(first)

    def test_append_slice_drop_boundary_inclusive_at_cutoff(self) -> None:
        """An existing row exactly at cutoff is dropped before concat.

        AC: boundary is >= cutoff (left-inclusive window [cutoff, …) means the
        cutoff row belongs to the new slice, not the old bronze).
        """
        # Arrange
        cutoff = datetime(2021, 1, 19)
        existing = _make_df([datetime(2021, 1, 18, 0), cutoff])  # cutoff row present
        new_slice = _make_df([cutoff, datetime(2021, 1, 19, 1)])

        # Act
        result = Ingestion.append_slice(existing, new_slice, cutoff=cutoff)

        # Assert — only 1 pre-cutoff row from existing + 2 new_slice rows = 3
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Steps 4–5: Ingestion.bronze_exists / load_bronze (I/O, tmp_path)
# ---------------------------------------------------------------------------


class TestBronzeIO:
    """bronze_exists and load_bronze work against a real tmp_path."""

    def _make_ingestion(
        self, tmp_path: Path, bronze_schema: dict[str, str]
    ) -> Ingestion:
        from tests.conftest import _FULL_CONFIG_DICT

        base = AppConfig.model_validate(_FULL_CONFIG_DICT)
        bronze_path = tmp_path / "bronze.parquet"
        config = base.model_copy(
            update={
                "data": base.data.model_copy(
                    update={
                        "raw_path": str(tmp_path / "raw"),
                        "bronze_path": str(bronze_path),
                    }
                ),
                "ingest": base.ingest.model_copy(
                    update={"bronze_schema": bronze_schema}
                ),
            }
        )
        return Ingestion(config=config)

    def test_bronze_exists_false_when_absent(
        self, tmp_path: Path, bronze_schema: dict[str, str]
    ) -> None:
        """bronze_exists() returns False when the bronze file does not exist."""
        # Arrange
        ingestion = self._make_ingestion(tmp_path, bronze_schema)

        # Act / Assert
        assert ingestion.bronze_exists() is False

    def test_bronze_exists_true_after_write(
        self, tmp_path: Path, bronze_schema: dict[str, str]
    ) -> None:
        """bronze_exists() returns True after save_bronze writes the file."""
        # Arrange
        ingestion = self._make_ingestion(tmp_path, bronze_schema)
        df = _make_df([datetime(2021, 1, 18)])

        # Act
        ingestion.save_bronze(df)

        # Assert
        assert ingestion.bronze_exists() is True

    def test_load_bronze_round_trips_written_frame(
        self, tmp_path: Path, bronze_schema: dict[str, str]
    ) -> None:
        """load_bronze() returns a DataFrame with the same rowcount that was saved."""
        # Arrange
        ingestion = self._make_ingestion(tmp_path, bronze_schema)
        df = _make_df([datetime(2021, 1, 18), datetime(2021, 1, 19)])

        # Act
        ingestion.save_bronze(df)
        loaded = ingestion.load_bronze()

        # Assert
        assert len(loaded) == len(df)


# ---------------------------------------------------------------------------
# Steps 6–7: load_raw(since=...) — synthetic month folders under tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture
def three_month_raw_tree(tmp_path: Path) -> Path:
    """Three months of tiny daily parquets with distinct sentinel timestamps.

    month 1: rows in 2021-01, month 2: rows in 2021-02, month 3: rows in 2021-03.
    """
    raw_root = tmp_path / "raw"
    for month_tag, start in [
        ("2021_01", pd.Timestamp("2021-01-01")),
        ("2021_02", pd.Timestamp("2021-02-01")),
        ("2021_03", pd.Timestamp("2021-03-01")),
    ]:
        month_dir = raw_root / month_tag
        month_dir.mkdir(parents=True, exist_ok=True)
        _write_daily_parquet(month_dir / f"{month_tag}_01.parquet", start, n_rows=2)
    return raw_root


def _ingestion_for_since(
    tmp_path: Path, raw_root: Path, bronze_schema: dict[str, str]
) -> Ingestion:
    from tests.conftest import _FULL_CONFIG_DICT

    base = AppConfig.model_validate(_FULL_CONFIG_DICT)
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "raw_path": str(raw_root),
                    "bronze_path": str(tmp_path / "bronze.parquet"),
                }
            ),
            "ingest": base.ingest.model_copy(
                update={
                    "start_year_month": "2021-01",
                    "end_year_month": "2021-03",
                    "bronze_schema": bronze_schema,
                }
            ),
        }
    )
    return Ingestion(config=config)


class TestLoadRawSince:
    """load_raw(since=...) reads only months at-or-after the given datetime."""

    def test_load_raw_since_reads_only_month_at_or_after(
        self, tmp_path: Path, three_month_raw_tree: Path, bronze_schema: dict[str, str]
    ) -> None:
        """since=2021-02-01 reads months 2 and 3 only; month-1 rows absent.

        AC: incremental reads skip already-ingested raw months.
        """
        # Arrange
        ingestion = _ingestion_for_since(tmp_path, three_month_raw_tree, bronze_schema)
        since = datetime(2021, 2, 1)

        # Act
        df = ingestion.load_raw(since=since)

        # Assert — only months 2+3 present; month-1 sentinel (Jan) must be absent
        assert (df["tpep_pickup_datetime"] < pd.Timestamp("2021-02-01")).sum() == 0

    def test_load_raw_since_none_reads_full_range(
        self, tmp_path: Path, three_month_raw_tree: Path, bronze_schema: dict[str, str]
    ) -> None:
        """load_raw(since=None) returns all months — backward-compat regression guard.

        AC: existing callers (Ingestion.run(), bulk_advance.py) unaffected.
        """
        # Arrange
        ingestion = _ingestion_for_since(tmp_path, three_month_raw_tree, bronze_schema)

        # Act
        df = ingestion.load_raw(since=None)

        # Assert — all three months present (6 rows total from fixture)
        assert len(df) == 6
