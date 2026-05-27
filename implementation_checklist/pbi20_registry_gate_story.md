# PBI 20 — MLflow Model Registry + Champion-vs-Challenger Gate

---

## User Story

As a pipeline operator, I want the training pipeline to register each trained model in the MLflow Model Registry and automatically promote it to Production only when it beats the current champion's test WAPE by a configurable margin, so that the Production model slot always holds a validated improvement and regressions are blocked without manual intervention.

---

## Acceptance Criteria

### Config changes

**AC-1.** `config.yml` contains an `mlflow.tracking_uri` value of `sqlite:///mlflow.db` (replacing the previous file-store URI).

**AC-2.** `config.yml` contains `mlflow.model_name` with default value `ny_taxi_demand_model`.

**AC-3.** `config.yml` contains `train.gate.min_wape_lift_pct` with default value `2.0`.

### MLflow tracking URI migration

**AC-4.** Given the training pipeline runs, when `Trainer.__init__()` is called, then `mlflow.set_tracking_uri()` is called with the value from `mlflow.tracking_uri` in config (not a hardcoded path).

**AC-5.** Given `mlflow.tracking_uri` is `sqlite:///mlflow.db`, when `mlflow.register_model()` is invoked, the call succeeds without raising a `MlflowException` about unsupported registry operations.

### Model registration

**AC-6.** After each successful training run, a new model version is registered under the name given by `mlflow.model_name` in the MLflow Model Registry.

**AC-7.** The registered version's source artifact matches the URI of the `mlflow.lightgbm.log_model` call from within the same run.

### First-ever promotion

**AC-8.** Given no Production model version exists for the configured model name, when training completes successfully, then the new version is transitioned to stage `Production` without consulting the gate threshold.

### Champion-vs-challenger gate — accepted

**AC-9.** Given a Production model exists with a logged `model_test_wape`, when the challenger's `model_test_wape` yields a `wape_lift_pct` >= `train.gate.min_wape_lift_pct`, then the challenger version is transitioned to `Production` and the previous Production version is transitioned to `Archived`.

**AC-10.** Given a promotion occurs, `models/model.pkl` on disk is overwritten with the newly promoted model.

### Champion-vs-challenger gate — rejected

**AC-11.** Given a Production model exists, when the challenger's `wape_lift_pct` < `train.gate.min_wape_lift_pct`, then the challenger version is NOT transitioned to `Production` and the incumbent Production version remains unchanged.

**AC-12.** Given a gate rejection, the MLflow run for the challenger has a tag `gate_result` set to `rejected` and a tag `gate_rejection_reason` set to a non-empty string describing why (e.g., `"wape_lift_pct 1.4 < required 2.0"`).

**AC-13.** Given a gate rejection, `models/model.pkl` on disk is NOT overwritten.

### Baseline unavailable

**AC-14.** Given the Gold dataset contains no week-ago overlap (baseline metrics are `None`), when the gate evaluates the challenger against an existing Production champion, then the challenger is rejected with `gate_rejection_reason` set to `no_baseline_for_comparison` and `models/model.pkl` is NOT overwritten.

**AC-15.** Given no baseline metrics AND no Production model exists (first-ever run), the rule in AC-8 still applies: the model is auto-promoted regardless of baseline availability.

### Observability

**AC-16.** Whether accepted or rejected, the MLflow run always has a tag `gate_result` set to either `promoted` or `rejected`.

---

## Edge Cases to Consider

- The Production model's stored `model_test_wape` metric is missing or `NaN` (e.g., logged from an older run before this PBI shipped) — gate should reject with reason `champion_wape_unavailable` rather than raise an unhandled exception.
- `train.gate.min_wape_lift_pct` is absent from `config.yml` — code must fall back to `2.0` without crashing.
- `mlflow.model_name` is absent from `config.yml` — code must fall back to `ny_taxi_demand_model`.
- A model version exists in the registry under the configured name but none has stage `Production` yet (e.g., prior runs all rejected) — treated the same as first-ever: auto-promote.
- Concurrent training runs attempting to register and promote simultaneously — MLflow's registry does not guarantee atomicity; this is a known risk but not required to be solved in this PBI.
- The SQLite file (`mlflow.db`) does not exist on first run — MLflow creates it automatically; no pre-migration step needed, but this should be confirmed in the smoke test.
- `wape_lift_pct` is exactly equal to `min_wape_lift_pct` — the gate must treat this as a pass (>= not >).

---

## Out of Scope

- Rewiring `Predictor.load_model()` to pull from the registry at inference time (PBI 21).
- Switching from MLflow stage names (`Production`, `Archived`) to MLflow 3.x aliases — deferred; stages work in MLflow >= 3.11.1 under backward compatibility.
- Provisioning or configuring a remote MLflow tracking server; `sqlite:///mlflow.db` is a local file.
- Pruning or deleting old non-Production model versions from the registry.
- Exposing gate outcome via the FastAPI `/heatmap` or any other HTTP endpoint.
- Any change to `app/api/main.py` or the serving path.
- UI or notification when a promotion is blocked.

---

## Dependencies

- **PBI 18 (shipped)** — `app/core/metrics.py` with `compute_wape_lift()` and the `model_test_wape` / `baseline_test_wape` logging conventions are already in place. The gate reads `model_test_wape` directly.

---

## Open Questions

- **Champion WAPE retrieval method**: the gate needs to read the incumbent Production model's `model_test_wape`. The two candidate approaches are (a) query the MLflow tracking store for the run linked to the Production model version, or (b) log the metric as a registered model tag at promotion time for fast lookup. Which approach is preferred? (Approach b is simpler; approach a avoids duplicating data.)
- **Rejection disk-preservation scope**: AC-13 specifies that `models/model.pkl` is not overwritten on rejection. Should the `save_model()` call be skipped entirely, or should the challenger be saved to a separate path (e.g., `models/challenger.pkl`) for manual inspection? The current spec says skip entirely — confirm this is acceptable.
- **SQLite path**: `sqlite:///mlflow.db` resolves relative to the working directory at runtime. Should this be an absolute path (e.g., `sqlite:////home/jason/NewYorkTaxiDemandForecasting/mlflow.db`) in `config.yml`, or is a relative path acceptable given the pipeline is always run from the project root?
