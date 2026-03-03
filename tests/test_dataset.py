import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from steps.ingest import Ingestion


@pytest.fixture
def config_data():
    return {
        'data': {
            'train_path': 'data/train.parquet',
            'test_path': 'data/test.parquet',
        }
    }


@pytest.fixture
def sample_data():
    train = pd.DataFrame({'hour': [8, 9], 'trip_count': [100, 150]})
    test = pd.DataFrame({'hour': [10], 'trip_count': [120]})
    return train, test


@patch("builtins.open", new_callable=mock_open, read_data="dummy")
@patch("yaml.safe_load")
@patch("pandas.read_parquet")
def test_load_data(mock_read_parquet, mock_safe_load, mock_open_file, config_data, sample_data):
    mock_safe_load.return_value = config_data
    mock_read_parquet.side_effect = sample_data

    ingestion = Ingestion()
    train_data, test_data = ingestion.load_data()

    pd.testing.assert_frame_equal(train_data, sample_data[0])
    pd.testing.assert_frame_equal(test_data, sample_data[1])

    mock_read_parquet.assert_any_call('data/train.parquet')
    mock_read_parquet.assert_any_call('data/test.parquet')
