"""Run data ingestion for the configured date range (Bronze layer).

Usage:
    uv run python run_ingest.py
"""

from src.steps.ingest import Ingestion

if __name__ == "__main__":
    ingestor = Ingestion()
    start = ingestor.config["ingest"]["start_year_month"]
    end = ingestor.config["ingest"]["end_year_month"]
    print(f"Fetching NYC Yellow Taxi data from {start} to {end} ...")
    paths = ingestor.fetch_data()
    print(f"\nDone! Files saved to:\n" + "\n".join(f"  {p}" for p in paths))
