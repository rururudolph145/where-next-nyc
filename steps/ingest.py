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

    def _fetch_to_parquet(self, where_clause: str, output_path: str) -> int:
        """Paginate through the API using *where_clause* and write rows to *output_path*.

        Args:
            where_clause: SoQL $where expression that filters the rows to fetch.
            output_path: Destination parquet file path.

        Returns:
            Total number of rows written (0 if the query returned no results).
        """
        writer: pq.ParquetWriter | None = None
        offset = 0
        total_rows = 0

        try:
            while True:
                params: dict[str, str | int] = {
                    "$where" : where_clause,
                    "$limit" : PAGE_SIZE,
                    "$offset": offset,
                    "$order" : "tpep_pickup_datetime",
                }
                resp = requests.get(BASE_URL, params=params, timeout=60)
                resp.raise_for_status()

                page: list[dict[str, object]] = resp.json()
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
                print(f"Saved → {output_path}  ({total_rows:,} rows)")

        return total_rows

    def fetch_data(self) -> str:
        """Download raw trip data for the configured date range and write to bronze parquet.

        Returns:
            Output path of the written parquet file.
        """
        os.makedirs("data/bronze", exist_ok=True)
        output_path = f"data/bronze/yellow_tripdata_{START_DATE}_{END_DATE}.parquet"
        where_clause = (
            f"tpep_pickup_datetime >= '{START_DATE}T00:00:00' "
            f"AND tpep_pickup_datetime < '{END_EXCLUSIVE}T00:00:00'"
        )
        self._fetch_to_parquet(where_clause, output_path)
        return output_path

    def fetch_day(self, date_str: str) -> str:
        """Download raw trip data for a single calendar day and write to bronze parquet.

        Args:
            date_str: Calendar day in ISO format (e.g. ``"2022-01-01"``).

        Returns:
            Output path of the written parquet file. The file will be absent if
            the API returned no rows for that day.
        """
        from datetime import date, timedelta

        next_day = (date.fromisoformat(date_str) + timedelta(days=1)).isoformat()
        os.makedirs("data/bronze", exist_ok=True)
        output_path = f"data/bronze/yellow_tripdata_{date_str}.parquet"
        where_clause = (
            f"tpep_pickup_datetime >= '{date_str}T00:00:00' "
            f"AND tpep_pickup_datetime < '{next_day}T00:00:00'"
        )
        self._fetch_to_parquet(where_clause, output_path)
        return output_path

    def load_data(self):
        train_path = self.config['data']['train_path']
        test_path  = self.config['data']['test_path']
        train = pd.read_parquet(train_path)
        test  = pd.read_parquet(test_path)
        return train, test
