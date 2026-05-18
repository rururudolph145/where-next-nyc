"""Ingestion library code (Bronze → Silver → Gold).

Modules:
    ingest:   Raw → Bronze (Socrata API → parquet).
    clean:    Bronze → Silver (validation, cleaning).
    feature:  Silver → Gold (feature engineering).

These modules are called by the orchestration runners under
``pipeline/`` at the repo root.
"""
