"""Custom exceptions for the inference module."""

from __future__ import annotations


class NoApprovedModelError(Exception):
    """Raised when no approved model version exists in the MLflow registry.

    Args:
        model_name: Name of the MLflow registered model.
        stages_checked: List of MLflow stages that were queried.
    """

    def __init__(self, model_name: str, stages_checked: list[str]) -> None:
        stages_str = ", ".join(stages_checked)
        message = (
            f"No approved model found for '{model_name}'. "
            f"Checked stages: [{stages_str}]. "
            "Register a Production or Archived version before running the simulator."
        )
        super().__init__(message)
        self.model_name = model_name
        self.stages_checked = stages_checked
