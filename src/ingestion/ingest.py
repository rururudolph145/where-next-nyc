"""Data ingestion from the NYC TLC Socrata API into the Bronze layer."""

import calendar
import os
import shutil
import tempfile
from datetime import date, timedelta
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.shared.config import load_config

SOCRATA_CATALOG_URL = "https://api.us.socrata.com/api/catalog/v1"
SOCRATA_DOMAIN = "data.cityofnewyork.us"
RESOURCE_URL_TEMPLATE = "https://data.cityofnewyork.us/resource/{dataset_id}.json"


def resolve_dataset_id(year: int) -> str:
    """Look up the Socrata dataset ID for a given year's Yellow Taxi Trip Data.

    Queries the Socrata Discovery API for datasets matching
    ``"<year> Yellow Taxi Trip Data"`` on the NYC Open Data domain.

    Args:
        year: Four-digit year (e.g. 2021).

    Returns:
        The Socrata dataset identifier (e.g. ``"m6nq-qud6"``).

    Raises:
        ValueError: If no matching dataset is found for the given year.
    """
    resp = requests.get(
        SOCRATA_CATALOG_URL,
        params={
            "q": f"{year} Yellow Taxi Trip Data",
            "domains": SOCRATA_DOMAIN,
            "limit": 5,
        },
        timeout=15,
    )
    resp.raise_for_status()

    for result in resp.json().get("results", []):
        name: str = result.get("resource", {}).get("name", "")
        if str(year) in name and "yellow taxi" in name.lower():
            dataset_id: str = result["resource"]["id"]
            return dataset_id

    raise ValueError(
        f"No Socrata dataset found for {year} Yellow Taxi Trip Data. "
        f"Check https://data.cityofnewyork.us for available years."
    )


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

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise Ingestion with configuration.

        Args:
            config: Parsed configuration dictionary. If ``None``, loads
                from ``config.yml`` via :func:`src.shared.config.load_config`.
        """
        self.config = config or load_config()
        self._session = _make_session()

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

    def _fetch_to_parquet(
        self, base_url: str, where_clause: str, output_path: str
    ) -> int:
        """Paginate through the API and write rows to *output_path*.

        Writes to a temporary file first and renames it atomically on success,
        so a partial file is never left at *output_path* if an error occurs
        mid-pagination.

        Args:
            base_url: Socrata JSON API endpoint for the target dataset.
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
                    base_url, params=params, timeout=REQUEST_TIMEOUT_SECONDS
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
                    f"  offset={offset:>7,}  batch={len(batch):,}"
                    f"  total={total_rows:,}"
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

    def fetch_data(self) -> list[str]:
        """Download raw trip data for the configured month range into the bronze layer.

        Reads ``ingest.start_year_month`` and ``ingest.end_year_month`` from
        the configuration and fetches each month via :meth:`fetch_month`.

        Returns:
            List of output parquet file paths (one per month).
        """
        start_ym: str = self.config["ingest"]["start_year_month"]
        end_ym: str = self.config["ingest"]["end_year_month"]
        s_year, s_month = (int(x) for x in start_ym.split("-"))
        e_year, e_month = (int(x) for x in end_ym.split("-"))

        paths: list[str] = []
        year, month = s_year, s_month
        while (year, month) <= (e_year, e_month):
            paths.append(self.fetch_month(year, month))
            month += 1
            if month > 12:
                month = 1
                year += 1
        return paths

    def fetch_day(self, date_str: str) -> str:
        """Download raw trip data for a single calendar day to bronze parquet.

        The file is saved under the configured ``data.bronze_path`` in a
        ``YYYY_MM/`` subfolder as ``YYYY_MM_DD.parquet``.

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

        bronze_base: str = self.config["data"]["bronze_path"]
        tag = f"{parsed.year}_{parsed.month:02d}"
        out_dir = f"{bronze_base}/{tag}"
        os.makedirs(out_dir, exist_ok=True)
        output_path = (
            f"{out_dir}/{tag}_{parsed.day:02d}.parquet"
        )

        next_day = (parsed + timedelta(days=1)).isoformat()
        dataset_id = resolve_dataset_id(parsed.year)
        base_url = RESOURCE_URL_TEMPLATE.format(dataset_id=dataset_id)

        where_clause = (
            f"tpep_pickup_datetime >= '{date_str}T00:00:00' "
            f"AND tpep_pickup_datetime < '{next_day}T00:00:00'"
        )
        rows_written = self._fetch_to_parquet(base_url, where_clause, output_path)
        if rows_written == 0:
            print(
                f"  Warning: no rows returned for {date_str}; "
                "bronze file not written."
            )
        return output_path

    def fetch_month(self, year: int, month: int) -> str:
        """Download raw trip data for a full calendar month into the bronze layer.

        The file is saved under the configured ``data.bronze_path`` as
        ``YYYY_MM/YYYY_MM.parquet``.
        If the file already exists the download is skipped (idempotent).

        Args:
            year: Four-digit year (e.g. 2022).
            month: Month number (1-12).

        Returns:
            Output path of the written (or already existing) parquet file.
        """
        bronze_base: str = self.config["data"]["bronze_path"]
        tag = f"{year}_{month:02d}"
        out_dir = f"{bronze_base}/{tag}"
        os.makedirs(out_dir, exist_ok=True)
        output_path = f"{out_dir}/{tag}.parquet"

        if os.path.exists(output_path):
            print(f"Skipping {tag} — {output_path} already exists.")
            return output_path

        first_day = date(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        next_month_first = first_day + timedelta(days=last_day)

        dataset_id = resolve_dataset_id(year)
        base_url = RESOURCE_URL_TEMPLATE.format(dataset_id=dataset_id)
        print(f"  Resolved dataset for {year}: {dataset_id}")

        where_clause = (
            f"tpep_pickup_datetime >= '{first_day.isoformat()}T00:00:00' "
            f"AND tpep_pickup_datetime < "
            f"'{next_month_first.isoformat()}T00:00:00'"
        )
        rows_written = self._fetch_to_parquet(base_url, where_clause, output_path)
        if rows_written == 0:
            print(
                f"  Warning: no rows returned for {tag}; bronze file not written."
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
