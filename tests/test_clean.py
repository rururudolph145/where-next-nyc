import pytest
import pandas as pd
from src.ingestion.clean import DataCleaner


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2023-01-01 08:00:00",
                "2023-01-01 09:00:00",
                None,
            ],
            "tpep_dropoff_datetime": [
                "2023-01-01 08:15:00",
                "2023-01-01 09:20:00",
                None,
            ],
            "passenger_count": [2, 1, 0],
            "trip_distance": [1.5, 3.2, 0.0],
            "fare_amount": [7.5, 12.0, 5.0],
            "PULocationID": [1, 2, 3],
        }
    )


@pytest.fixture
def cleaner():
    config = {
        "env": "local",
        "data": {
            "bronze_path": "data/bronze/test.parquet",
            "silver_path": "data/silver/test.parquet",
        },
        "clean": {"null_strategy": "drop"},
        "inference": {"simulation": {"start_hour": "2023-06-01 00:00:00"}},
        "model": {"store_path": "models/"},
    }
    return DataCleaner(config)


def test_clean_data(sample_data, cleaner):
    result, bounds = cleaner.clean(sample_data)

    # Rows with null datetimes or non-positive passenger/distance should be dropped
    assert result["tpep_pickup_datetime"].notna().all()
    assert (result["passenger_count"] > 0).all()
    assert (result["trip_distance"] > 0).all()

    # Time features should be created
    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns

    # Bounds should be returned
    assert "q1" in bounds
    assert "q3" in bounds
    assert "iqr" in bounds
    assert bounds["iqr"] == bounds["q3"] - bounds["q1"]


def test_clean_transform_mode(cleaner):
    """Verify that provided IQR bounds are used instead of recomputed."""
    data = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2023-01-01 08:00:00",
                "2023-01-01 09:00:00",
                "2023-01-01 10:00:00",
            ],
            "tpep_dropoff_datetime": [
                "2023-01-01 08:15:00",
                "2023-01-01 09:20:00",
                "2023-01-01 10:30:00",
            ],
            "passenger_count": [1, 1, 1],
            "trip_distance": [2.0, 3.0, 4.0],
            "fare_amount": [10.0, 20.0, 100.0],
            "PULocationID": [1, 2, 3],
        }
    )

    # Fit mode — bounds computed from data
    _, fit_bounds = cleaner.clean(data)

    # Transform mode — use tight bounds that exclude the 100.0 fare
    tight_bounds = {"q1": 10.0, "q3": 20.0, "iqr": 10.0}
    result, returned_bounds = cleaner.clean(data, iqr_bounds=tight_bounds)

    # The returned bounds should be exactly what we passed in
    assert returned_bounds == tight_bounds

    # The 100.0 fare row should be filtered out with tight bounds
    assert (result["fare_amount"] <= 20.0 + 1.5 * 10.0).all()
    assert 100.0 not in result["fare_amount"].values
