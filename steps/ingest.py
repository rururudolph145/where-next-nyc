import os
import yaml
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

START_DATE    = "2022-01-01"
END_DATE      = "2022-01-14"
END_EXCLUSIVE = "2022-01-15"

DATASET_ID = "qp3b-zxtp"  # 2022 Yellow Taxi Trip Data
BASE_URL   = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
PAGE_SIZE  = 50_000

DATETIME_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
FLOAT_COLS    = ["trip_distance", "fare_amount", "extra", "mta_tax",
                 "tip_amount", "tolls_amount", "improvement_surcharge", "total_amount"]
INT_COLS      = ["vendorid", "passenger_count", "ratecodeid",
                 "payment_type", "pulocationid", "dolocationid"]


class Ingestion:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        with open('config.yml', 'r') as f:
            return yaml.safe_load(f)

    def cast_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in DATETIME_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in FLOAT_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        for col in INT_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")
        return df

    def fetch_data(self) -> str:
        """Download raw trip data from the NYC Open Data API and write to parquet.
        Returns the output path."""
        os.makedirs("data/bronze", exist_ok=True)
        output_path = f"data/bronze/yellow_tripdata_{START_DATE}_{END_DATE}.parquet"

        where_clause = (
            f"tpep_pickup_datetime >= '{START_DATE}T00:00:00' "
            f"AND tpep_pickup_datetime < '{END_EXCLUSIVE}T00:00:00'"
        )

        writer     = None
        offset     = 0
        total_rows = 0

        try:
            while True:
                params = {
                    "$where" : where_clause,
                    "$limit" : PAGE_SIZE,
                    "$offset": offset,
                    "$order" : "tpep_pickup_datetime",
                }
                resp = requests.get(BASE_URL, params=params, timeout=60)
                resp.raise_for_status()

                page = resp.json()
                if not page:
                    break

                batch = self.cast_batch(pd.DataFrame(page))
                table = pa.Table.from_pandas(batch, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")

                writer.write_table(table)
                total_rows += len(batch)
                print(f"  offset={offset:>7,}  batch={len(batch):,}  total={total_rows:,}")

                if len(page) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

        finally:
            if writer:
                writer.close()
                print(f"\nSaved → {output_path}  ({total_rows:,} rows)")

        return output_path

    def load_data(self):
        train_path = self.config['data']['train_path']
        test_path  = self.config['data']['test_path']
        train = pd.read_parquet(train_path)
        test  = pd.read_parquet(test_path)
        return train, test
