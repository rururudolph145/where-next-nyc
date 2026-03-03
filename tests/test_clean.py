import pytest
import pandas as pd
from steps.clean import Cleaner


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'tpep_pickup_datetime': ['2023-01-01 08:00:00', '2023-01-01 09:00:00', None],
        'tpep_dropoff_datetime': ['2023-01-01 08:15:00', '2023-01-01 09:20:00', None],
        'passenger_count': [2, 1, 0],
        'trip_distance': [1.5, 3.2, 0.0],
        'fare_amount': [7.5, 12.0, 5.0],
        'PULocationID': [1, 2, 3],
    })


@pytest.fixture
def cleaner():
    return Cleaner()


def test_clean_data(sample_data, cleaner):
    result = cleaner.clean_data(sample_data)

    # Rows with null datetimes or non-positive passenger/distance should be dropped
    assert result['tpep_pickup_datetime'].notna().all()
    assert (result['passenger_count'] > 0).all()
    assert (result['trip_distance'] > 0).all()

    # Time features should be created
    assert 'hour' in result.columns
    assert 'day_of_week' in result.columns
    assert 'month' in result.columns
    assert 'is_weekend' in result.columns
