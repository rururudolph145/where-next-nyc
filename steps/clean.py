import pandas as pd


class Cleaner:
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean raw NY taxi trip data and engineer time-based features."""
        data = data.copy()

        # Drop rows with missing pickup/dropoff times
        data.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)

        # Parse datetimes
        data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
        data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])

        # Remove trips with non-positive passenger count or trip distance
        data = data[(data['passenger_count'] > 0) & (data['trip_distance'] > 0)]

        # Remove fare outliers using IQR
        Q1 = data['fare_amount'].quantile(0.25)
        Q3 = data['fare_amount'].quantile(0.75)
        IQR = Q3 - Q1
        data = data[(data['fare_amount'] >= Q1 - 1.5 * IQR) & (data['fare_amount'] <= Q3 + 1.5 * IQR)]

        # Feature engineering
        data['hour'] = data['tpep_pickup_datetime'].dt.hour
        data['day_of_week'] = data['tpep_pickup_datetime'].dt.dayofweek
        data['month'] = data['tpep_pickup_datetime'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

        return data
