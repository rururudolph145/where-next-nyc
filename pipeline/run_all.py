"""Run the full pipeline: ingest → clean → feature engineering → train.

Usage:
    uv run python -m pipeline.run_all           # full pipeline
    uv run python pipeline/01_ingest.py         # just ingestion
    uv run python pipeline/02_clean.py          # just cleaning
    uv run python pipeline/03_feature.py        # just feature engineering
    uv run python pipeline/04_train.py          # just training
"""

import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

STEPS = [
    ("01_ingest", "run_ingestion"),
    ("02_clean", "run_clean"),
    ("03_feature", "run_feature"),
    ("04_train", "run_train"),
]


def run_all() -> None:
    """Execute every pipeline step in order."""
    logging.info("=== Starting full pipeline ===")
    for module_name, func_name in STEPS:
        mod = importlib.import_module(f"pipeline.{module_name}")
        func = getattr(mod, func_name)
        func()
    logging.info("=== Full pipeline complete ===")


if __name__ == "__main__":
    run_all()
