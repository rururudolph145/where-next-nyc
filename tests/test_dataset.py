from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from unittest.mock import mock_open, patch

from app.ingestion.ingest import Ingestion


@pytest.fixture
def config_data():
    return {
        "data": {
            "train_path": "data/train.parquet",
            "test_path": "data/test.parquet",
        }
    }


@pytest.fixture
def sample_data():
    train = pd.DataFrame({"hour": [8, 9], "trip_count": [100, 150]})
    test = pd.DataFrame({"hour": [10], "trip_count": [120]})
    return train, test


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
    config: dict[str, Any] = {
        "data": {
            "raw_path": str(ordered_raw_tree),
            "bronze_path": str(bronze_path),
        },
        "ingest": {
            "start_year_month": "2021-01",
            "end_year_month": "2021-02",
            "bronze_schema": bronze_schema,
        },
    }

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
    config: dict[str, Any] = {
        "data": {
            "raw_path": str(out_of_order_raw_tree),
            "bronze_path": str(bronze_path),
        },
        "ingest": {
            "start_year_month": "2021-01",
            "end_year_month": "2021-01",
            "bronze_schema": bronze_schema,
        },
    }

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
