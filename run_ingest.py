"""Run data ingestion for the configured date range (Bronze layer).

Usage:
    uv run python run_ingest.py
"""

from steps.ingest import Ingestion, START_DATE, END_DATE

if __name__ == "__main__":
    print(f"Fetching NYC Yellow Taxi data from {START_DATE} to {END_DATE} ...")
    ingestor = Ingestion()
    path = ingestor.fetch_data()
    print(f"\nDone! File saved to: {path}")
