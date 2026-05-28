"""Champion-vs-challenger promotion gate for the MLflow Model Registry.

Extracted from ``app/training/trainer.py`` to keep ``Trainer.run()`` within
the 50-line guideline and to allow independent unit testing.
"""

from __future__ import annotations

from mlflow.tracking import MlflowClient

from app.core.metrics import compute_wape_lift


def apply_gate(
    client: MlflowClient,
    model_name: str,
    new_version: str,
    challenger_wape: float,
    min_lift_pct: float,
) -> tuple[str, str | None]:
    """Evaluate a challenger model against the current Production champion.

    Decision tree:
    - No Production version exists → auto-promote (first-ever or all prior rejected).
    - Champion tag ``model_test_wape`` missing or ``'nan'`` → reject with reason
      ``champion_wape_unavailable``.
    - Challenger lift >= *min_lift_pct* → promote challenger; archive champion.
    - Otherwise → reject with a descriptive reason string.

    The caller is responsible for:
    - Setting ``model_test_wape`` on *new_version* before calling this function.
    - Writing the returned ``gate_result`` / ``gate_rejection_reason`` MLflow tags.
    - Calling ``save_model()`` only when the returned result is ``"promoted"``.

    Args:
        client: Initialised :class:`mlflow.tracking.MlflowClient` instance.
        model_name: Registered model name in the MLflow Model Registry.
        new_version: Version string of the newly registered challenger model.
        challenger_wape: Test WAPE of the challenger (from ``model_test_wape``
            metric logged during this run).
        min_lift_pct: Minimum WAPE lift percentage required to promote.

    Returns:
        A ``(gate_result, rejection_reason)`` tuple where *gate_result* is
        ``"promoted"`` or ``"rejected"`` and *rejection_reason* is a non-empty
        string on rejection, ``None`` on promotion.
    """
    production_versions = client.get_latest_versions(model_name, stages=["Production"])

    if not production_versions:
        # First-ever run or all prior versions rejected — auto-promote.
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production",
        )
        return ("promoted", None)

    champion = production_versions[0]
    raw_tag = champion.tags.get("model_test_wape")

    if raw_tag is None or raw_tag == "nan":
        return ("rejected", "champion_wape_unavailable")

    champion_wape = float(raw_tag)
    lift = compute_wape_lift(champion_wape, challenger_wape)

    if lift >= min_lift_pct:
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production",
        )
        client.transition_model_version_stage(
            name=model_name,
            version=champion.version,
            stage="Archived",
        )
        return ("promoted", None)

    reason = f"wape_lift_{lift:.2f}_below_threshold_{min_lift_pct}"
    return ("rejected", reason)
