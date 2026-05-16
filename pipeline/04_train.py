"""Step 4 — Training: Gold → Model (with MLflow logging)."""

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.config import load_config
from src.training.train import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def run_train() -> None:
    """Train the demand prediction model from Gold layer data.

    Performs time-based splitting, trains LGBMRegressor, evaluates on
    val/test, logs metrics to MLflow, and saves the model artifact.
    """
    config: dict[str, Any] = load_config()
    trainer = Trainer(config)

    logging.info("Training demand prediction model")
    result = trainer.run()
    logging.info(
        "Training complete. Val MAE=%.2f, Test MAE=%.2f. Model → %s",
        result["val_metrics"]["mae"],
        result["test_metrics"]["mae"],
        result["model_path"],
    )


if __name__ == "__main__":
    run_train()
