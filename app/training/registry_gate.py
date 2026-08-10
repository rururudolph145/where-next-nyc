"""Champion-vs-challenger promotion gate for the MLflow Model Registry.

Extracted from ``app/training/trainer.py`` to keep ``Trainer.run()`` within
the 50-line guideline and to allow independent unit testing.
"""

from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from app.core.metrics import compute_wape_lift


def apply_gate(
    client: MlflowClient,
    model_name: str,
    new_version: str,
    challenger_wape: float,
    min_lift_pct: float,
) -> tuple[str, str | None]:
    """Evaluate a challenger model against the current @champion alias.

    Decision tree:
    - No @champion alias exists → auto-promote (first-ever or all prior rejected).
    - Champion tag ``model_test_wape`` missing or ``'nan'`` → reject with reason
      ``champion_wape_unavailable``.
    - Challenger lift >= *min_lift_pct* → set @champion alias to new_version.
    - Otherwise → leave @champion alias untouched; return rejected.

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
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        champion = None

    if champion is None:
        # First-ever run or all prior versions rejected — auto-promote.
        client.set_registered_model_alias(model_name, "champion", new_version)
        return ("promoted", None)

    raw_tag = champion.tags.get("model_test_wape")

    if raw_tag is None or raw_tag == "nan":
        return ("rejected", "champion_wape_unavailable")

    champion_wape = float(raw_tag)
    lift = compute_wape_lift(champion_wape, challenger_wape)

    if lift >= min_lift_pct:
        client.set_registered_model_alias(model_name, "champion", new_version)
        return ("promoted", None)

    reason = f"wape_lift_{lift:.2f}_below_threshold_{min_lift_pct}"
    return ("rejected", reason)
