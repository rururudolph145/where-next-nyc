"""Step 5 — Inference: Trained Model + Gold → Predictions Table."""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_config
from app.inference.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def run_inference() -> None:
    """Run batch inference to generate demand predictions.

    Loads the trained model and Gold layer data, generates predictions,
    evaluates metrics, and saves the predictions table to parquet.
    """
    config: dict[str, Any] = load_config()
    predictor = Predictor(config)

    logging.info("Running batch inference")
    result = predictor.run()
    logging.info(
        "Inference complete. MAE=%.2f, RMSE=%.2f. %d predictions → %s",
        result["metrics"]["mae"],
        result["metrics"]["rmse"],
        result["n_predictions"],
        result["output_path"],
    )


if __name__ == "__main__":
    run_inference()
