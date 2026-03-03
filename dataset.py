import pandas as pd
import os

def extract_data():
    """Download and split NY Yellow Taxi trip data into train/test sets."""
    if not os.path.exists("data"):
        os.mkdir("data")

    # TODO: Replace with actual data download logic (e.g., NYC TLC parquet files)
    # Example: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
    raise NotImplementedError(
        "Implement data download from NYC TLC: "
        "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
    )

if __name__ == "__main__":
    extract_data()
