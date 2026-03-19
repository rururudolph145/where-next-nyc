import os
import shutil
import tempfile
from datetime import date, timedelta
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

START_DATE = "2022-01-01"
END_DATE = "2022-02-28"
END_EXCLUSIVE = "2022-03-01"

DATASET_ID = "qp3b-zxtp"  # 2022 Yellow Taxi Trip Data
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
PAGE_SIZE = 50_000
REQUEST_TIMEOUT_SECONDS = 60  # Socrata API can be slow on large offsets

DATETIME_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
FLOAT_COLS = [
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
]
INT_COLS = [
    "vendorid",
    "passenger_count",
    "ratecodeid",
    "payment_type",
    "pulocationid",
    "dolocationid",
]


def _make_session() -> requests.Session:
    """Create an HTTP session with automatic retry on transient failures.

    Returns:
        A ``requests.Session`` configured with exponential back-off and
        retries for HTTP 429 / 5xx responses.
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


class Ingestion:
    """Handles data ingestion from the NYC TLC Socrata API into the bronze layer.

    Fetch methods (``fetch_data``, ``fetch_day``) download raw trip records
    from the public Socrata API and write them as Snappy-compressed Parquet
    files under ``data/bronze/``.

    Loading methods (``load_data``) read pre-split parquet files from paths
    specified in ``config.yml``.
    """

    def __init__(self, config_path: str = "config.yml") -> None:
        """Initialise Ingestion and load configuration.

        Args:
            config_path: Path to the YAML configuration file. Defaults to
                ``"config.yml"`` relative to the current working directory.
        """
        self.config = self.load_config(config_path)
        self._session = _make_session()

    @staticmethod
    def load_config(config_path: str = "config.yml") -> dict[str, Any]:
        """Load and return the YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Parsed configuration as a nested dictionary.
        """
        with open(config_path) as f:
            return yaml.safe_load(f)

    def cast_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast DataFrame columns to their canonical types.

        Datetime columns are parsed with ``errors="coerce"`` so unparseable
        values become ``NaT``. Numeric columns are coerced similarly —
        invalid values become ``NaN`` / ``pd.NA`` rather than raising.

        Args:
            df: Raw DataFrame as returned by the Socrata JSON API.

        Returns:
            The same DataFrame with columns cast in-place.
        """
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

        Writes to a temporary file first and renames it atomically on success,
        so a partial file is never left at *output_path* if an error occurs
        mid-pagination.

        Args:
            where_clause: SoQL $where expression that filters the rows to fetch.
            output_path: Destination parquet file path.

        Returns:
            Total number of rows written (0 if the query returned no results).
        """
        writer: pq.ParquetWriter | None = None
        file_schema: pa.Schema | None = None
        offset = 0
        total_rows = 0
        success = False

        tmp_dir = os.path.dirname(output_path) or "."
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet.tmp", dir=tmp_dir)
        os.close(tmp_fd)

        try:
            while True:
                params: dict[str, str | int] = {
                    "$where": where_clause,
                    "$limit": PAGE_SIZE,
                    "$offset": offset,
                    "$order": "tpep_pickup_datetime",
                }
                resp = self._session.get(
                    BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
                )
                resp.raise_for_status()

                page: list[dict[str, object]] = resp.json()
                if not page:
                    break

                batch = self.cast_batch(pd.DataFrame(page))
                table = pa.Table.from_pandas(batch, preserve_index=False)

                if writer is None:
                    file_schema = table.schema
                    writer = pq.ParquetWriter(
                        tmp_path, file_schema, compression="snappy"
                    )

                # Normalise column order to match the schema from the first batch.
                assert file_schema is not None  # set above on first iteration
                table = table.select(file_schema.names).cast(file_schema)
                writer.write_table(table)
                total_rows += len(batch)
                print(
                    f"  offset={offset:>7,}  batch={len(batch):,}  total={total_rows:,}"
                )

                if len(page) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

            success = True

        finally:
            if writer:
                writer.close()

            if success:
                if total_rows > 0:
                    shutil.move(tmp_path, output_path)
                    print(f"Saved → {output_path}  ({total_rows:,} rows)")
                else:
                    os.unlink(tmp_path)
            else:
                # Exception path: remove the incomplete temp file.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        return total_rows

    # ── Fetch methods: download raw data from the Socrata API → Bronze ────────

    def fetch_data(self) -> str:
        """Download raw trip data for the configured date range and write to bronze parquet.

        Uses the module-level ``START_DATE`` / ``END_DATE`` / ``END_EXCLUSIVE``
        constants for the date range. ``config.yml`` is not consulted here; it
        is loaded only for use by ``load_data``.

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

        Raises:
            ValueError: If *date_str* is not a valid ISO-format date.
        """
        try:
            parsed = date.fromisoformat(date_str)
        except ValueError:
            raise ValueError(
                f"fetch_day: expected ISO date (YYYY-MM-DD), got {date_str!r}"
            ) from None

        next_day = (parsed + timedelta(days=1)).isoformat()
        os.makedirs("data/bronze", exist_ok=True)
        output_path = f"data/bronze/yellow_tripdata_{date_str}.parquet"
        where_clause = (
            f"tpep_pickup_datetime >= '{date_str}T00:00:00' "
            f"AND tpep_pickup_datetime < '{next_day}T00:00:00'"
        )
        rows_written = self._fetch_to_parquet(where_clause, output_path)
        if rows_written == 0:
            print(
                f"  Warning: no rows returned for {date_str}; bronze file not written."
            )
        return output_path

    # ── Load methods: read pre-split parquet files from disk ──────────────────

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the train and test parquet files specified in config.

        Returns:
            A ``(train, test)`` tuple of DataFrames.
        """
        train_path = self.config["data"]["train_path"]
        test_path = self.config["data"]["test_path"]
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        return train, test
