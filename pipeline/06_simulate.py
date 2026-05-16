"""Step 6 — Hourly inference simulation over out-of-sample period.

Runs the :class:`src.inference.simulate.HourlySimulator` to produce
per-zone, per-hour predictions for the configured simulation window
(default: March 2021).

Usage:
    uv run python pipeline/06_simulate.py
"""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.simulate import HourlySimulator
from src.shared.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def run_simulation() -> None:
    """Load config, run hourly simulation, and log results."""
    config: dict[str, Any] = load_config()

    sim_cfg = config.get("inference", {}).get("simulation", {})
    logging.info(
        "Simulation: %s → %s",
        sim_cfg.get("start_hour", "N/A"),
        sim_cfg.get("end_hour", "N/A"),
    )

    simulator = HourlySimulator(config)
    result = simulator.run()

    metrics: dict[str, float] = result["metrics"]
    logging.info(
        "Final metrics — MAE: %.2f | RMSE: %.2f | MAPE: %.2f%% | WAPE: %.2f%%",
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
        metrics["wape"],
    )
    logging.info(
        "Predictions: %d rows (%d hours) → %s",
        result["n_predictions"],
        result["n_hours"],
        result["output_path"],
    )


if __name__ == "__main__":
    run_simulation()
