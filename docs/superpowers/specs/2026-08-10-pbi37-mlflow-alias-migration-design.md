# PBI 37 — Fix Train→Inference Model Contract (MLflow Registry Aliases)

**Date:** 2026-08-10  
**Branch:** `feature/partitioned-prediction-history`  
**Scope:** Approach B — alias migration + sklearn flavour generalisation + path contract audit  
**Difficulty:** M  
**Vehicle:** feature-factory Lane C (pipeline-orchestrator → backend-builder)

---

## Problem

### Problem 1 — DAG inference ignores the MLflow registry entirely

The live inference DAG (`the_15_minute_refresh_inference_dag`) calls
`predict_helpers.run_predict()` → `Predictor.load_model()` → `joblib.load("data/models/model.pkl")`.
**The MLflow Model Registry is never consulted.** This breaks the operational contract:

- The trainer runs `mlflow.lightgbm.log_model()` + `mlflow.register_model()` to store the
  model artifact in the MLflow artifact store (under `mlruns/...`). The registry entry's
  `source` URI points to that artifact.
- On gate-promote, `trainer.py` also writes a disk copy (`joblib.dump → model.pkl`).
  The disk copy is a convenience artifact, not the source of truth.
- If you retrain, the gate rejects the challenger, and then you manually want to roll back to
  an earlier version by setting a different `@champion`, **`Predictor` would not see the change**.
  It would keep serving whatever `.pkl` happens to be on disk.

`simulator.py` already does the right thing — it calls `mlflow.lightgbm.load_model(uri)` via
the registry. `Predictor` must be brought to the same standard.

### Problem 2 — Deprecated MLflow stages API (13 call sites)

The codebase runs MLflow 3.11.1 but uses the MLflow 2.x `stages`-based API throughout:

- `get_latest_versions(stages=["Production"])` — deprecated since MLflow 2.9
- `transition_model_version_stage(stage=...)` — deprecated since MLflow 2.9

13 call sites across 4 files emit `FutureWarning` on every registry operation. In a future
MLflow release these methods will be removed.

### Problem 3 — Model flavour locked to LightGBM

`trainer.py` calls `mlflow.lightgbm.log_model()` and `simulator.py` / `predictor.py` call
`mlflow.lightgbm.load_model()`. This hard-codes LightGBM as the only permissible model class.
Adding `RandomForestRegressor` (or any other sklearn-API estimator) would require a second
code path.

`LGBMRegressor` is a fully valid scikit-learn estimator — it inherits `BaseEstimator` and
`RegressorMixin`. `mlflow.sklearn.log_model()` handles it natively, as it does
`RandomForestRegressor`. Switching to `mlflow.sklearn` is a 2-line change that removes the
lock and requires no structural changes to training or inference.

`pyfunc` would be the right choice if custom pre/post-processing needed to be baked into the
artifact, or if the model had no native MLflow flavour. Neither applies here — both planned
model types have the same `.predict(X)` sklearn contract. `pyfunc` would be overengineering.

### Problem 4 — Model path contract unlocked

The model path contract (which directory holds `model.pkl` + `iqr_bounds.json` shared between
the train DAG and the inference DAG) is correct via the `container_paths.py` override seam but
has no test locking it. A future refactor could break the contract invisibly.

---

## Goals

1. Replace all deprecated registry calls with the MLflow 3.x alias API.
2. Make `Predictor` registry-primary so rollback via `@champion` actually takes effect.
3. Replace `mlflow.lightgbm` flavour with `mlflow.sklearn` so any sklearn-API estimator works.
4. Lock the model path contract with a test.
5. Zero regressions — all existing tests pass.

---

## Out of Scope

- Bundling `iqr_bounds.json` into `model_bundle.pkl` (ADR-003 intent deferred — JSON alongside model.pkl is simpler at this scale).
- Renaming `config.yml`'s `model.store_path` from `data/models` → `data/models_trained`.
- Baseline-policy fallback ("last week same hour" if registry is down) — future PBI.
- `mlflow.db` migration from file store to SQLite (already using SQLite; file-store warning is a separate concern).
- Training a `RandomForestRegressor` end-to-end — this PBI only registers it in `MODELS` so config can select it. Hyperparameter tuning and comparison with LGBM is a future experiment.

---

## Design

### Section 1 — MLflow Alias Migration

**Semantic model:**

| Old concept     | New concept              | MLflow 3.x API                                     |
|-----------------|--------------------------|-----------------------------------------------------|
| `stage=Production` | alias `@champion`     | `set_registered_model_alias(name, "champion", ver)` |
| `stage=Archived`   | no alias (orphaned)   | no operation needed                                 |
| "first run"        | alias absent → auto-promote | `MlflowException(RESOURCE_DOES_NOT_EXIST)` caught |

The `@champion` alias is the single source of truth for "what version is currently serving."
Gate-reject leaves the existing `@champion` alias unchanged. There is no "archive" operation —
the prior champion version remains in the registry without an alias.

**`app/training/registry_gate.py` (`apply_gate`):**

```python
# Lookup: replace get_latest_versions(stages=["Production"])
try:
    champion_mv = client.get_model_version_by_alias(model_name, "champion")
except MlflowException:
    champion_mv = None  # no @champion set → auto-promote

# Promote: replace transition_model_version_stage
client.set_registered_model_alias(model_name, "champion", new_version)
```

The `gate_rejection_reason` MLflow tag write is unchanged. `NoApprovedModelError` is not
raised from `apply_gate` — that remains the caller's responsibility via `simulator.py`.

**`app/training/trainer.py`:**

Remove the inline `get_latest_versions` call at line 359 (duplicates `registry_gate` logic).
Delegate entirely to `apply_gate`. The `mlflow.register_model()` call is unchanged.

**`app/inference/simulator.py`:**

Replace:
```python
prod_versions = client.get_latest_versions(model_name, stages=["Production"])
arch_versions = client.get_latest_versions(model_name, stages=["Archived"])
```

With:
```python
try:
    prod_mv = client.get_model_version_by_alias(model_name, "champion")
except MlflowException:
    prod_mv = None
```

The Archived fallback is dropped — if `@champion` is absent, the flow falls through to the
disk fallback in `Predictor.load_model()`. `NoApprovedModelError` is still raised when
`@champion` is absent AND disk has no `model.pkl`.

**`app/api/main.py`:**

Both `get_latest_versions(stages=["Production"])` calls (at `/model-info` and `/health`)
replaced with `get_model_version_by_alias("champion")`.

**`app/inference/predictor.py` (`load_model`) — PRIMARY FIX:**

`Predictor.load_model()` currently loads from disk only (`joblib.load(model.pkl)`).
**The registry is never consulted.** This means `Predictor` (used by the live DAG) ignores
the champion alias — rollback via the registry has no effect on what gets served.

The fix makes the registry the primary source, matching what `simulator.py` already does:

```python
def load_model(self) -> Any:
    # Registry-primary: resolve @champion alias from the MLflow artifact store.
    # This is the operational path used by the live DAG.
    try:
        mlflow.set_tracking_uri(self.tracking_uri)
        model_uri = f"models:/{self.registry_model_name}@champion"
        self.model = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded model from registry: %s", model_uri)
        return self.model
    except MlflowException:
        pass  # alias absent or registry unreachable → disk fallback

    # Disk fallback for offline/standalone path (fresh clone, no mlflow.db).
    # The trainer writes this file whenever a model is promoted, so it is always
    # the most-recently-promoted model — an adequate offline substitute.
    model_path = os.path.join(self.store_path, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found in registry (@champion) or on disk ({model_path})"
        )
    self.model = joblib.load(model_path)
    logger.warning("Registry unavailable — loaded model from disk: %s", model_path)
    return self.model
```

The `tracking_uri` and `registry_model_name` are read from `self.config["mlflow"]`
(already wired into `Predictor.__init__` as part of this PBI).

The disk `model.pkl` is written by `trainer.save_model()` on every gate-promote, so it
always reflects the last promoted version. It is adequate for offline/demo use.
Rollback in production is achieved by setting a different `@champion` alias —
`Predictor` will resolve it on the next DAG tick.

---

### Section 1b — sklearn Flavour (Model-Type Generalisation)

**Why `mlflow.sklearn`, not `mlflow.pyfunc`:**

`LGBMRegressor` inherits `sklearn.base.BaseEstimator` and `RegressorMixin` — it is a
first-class sklearn estimator. `mlflow.sklearn` natively handles any object that satisfies
the sklearn estimator contract (`.fit()`, `.predict()`). `RandomForestRegressor` also
satisfies this contract. Switching to `mlflow.sklearn` requires changing exactly two call
sites and costs nothing at runtime — the serialisation format stays joblib/pickle under the
hood, and `predict()` calls are identical.

`mlflow.pyfunc` wraps a custom Python callable and is the right choice when:
- The model has no native MLflow flavour (e.g. a raw PyTorch module, a JAX function), or
- Pre/post-processing steps must be co-packaged inside the artifact.

Neither applies here. Using `pyfunc` for an sklearn estimator adds a wrapper layer with no
benefit — do not use it.

**`app/training/trainer.py`:**

```python
# was:
mlflow.lightgbm.log_model(self.model, artifact_path="model")
# becomes:
mlflow.sklearn.log_model(self.model, artifact_path="model")
```

The `MODELS` dict type annotation is loosened so any sklearn regressor can be registered:

```python
# was:
MODELS: dict[str, type[LGBMRegressor]] = {"LGBMRegressor": LGBMRegressor}

# becomes (import RegressorMixin from sklearn.base):
from sklearn.base import RegressorMixin
MODELS: dict[str, type[RegressorMixin]] = {
    "LGBMRegressor": LGBMRegressor,
    "RandomForestRegressor": RandomForestRegressor,
}
```

`self.model` annotation changes from `LGBMRegressor` to `RegressorMixin`.

**`app/inference/simulator.py` and `app/inference/predictor.py`:**

```python
# was:
mlflow.lightgbm.load_model(uri)
# becomes:
mlflow.sklearn.load_model(uri)
```

The disk fallback (`joblib.load(model.pkl)`) is unchanged — joblib serialises any Python
object regardless of framework.

**`app/training/trainer.py` — `save_model()` disk fallback:**

`joblib.dump(self.model, model_path)` is already framework-agnostic — no change needed.

**Config (`config.yml`):**

`model.name` stays `"LGBMRegressor"` (default). Switching to RF requires only:
```yaml
model:
  name: RandomForestRegressor
  params:
    n_estimators: 500
    random_state: 42
```
No code changes needed beyond this PBI.

---

### Section 2 — Model Path Contract Audit

**Current state:**

The `container_paths.py` override seam correctly unifies train and infer paths:
- `config.yml`: `model.store_path = "data/models"` (host default)
- `container_paths(config)["models"]` = `"/opt/airflow/data/models"` (container)
- Both `build_train_config()` and `build_predict_config()` override `model.store_path`
  with the same `paths["models"]` value → train and infer DAGs share the same directory.

No functional change required. The contract is locked with a new test file.

**New test file: `tests/test_container_paths.py`**

```
class TestContainerPaths:
    test_models_key_present          — "models" key exists in output dict
    test_models_path_under_prefix    — value starts with /opt/airflow
    test_models_path_not_absolute_in_config — raises ValueError when store_path starts with /
    test_train_infer_model_paths_agree  — build_train_config and build_predict_config
                                          produce equal model.store_path (@pytest.mark.integration)
```

**`orchestration/README.md` — mlflow.db reset note:**

Add a "Container MLflow Setup" subsection documenting the requirement to delete
`data/mlflow.db` before the first container training run (when an existing `mlflow.db`
was created on the host, its `artifact_location` column is a host absolute path
unreachable inside the container).

---

## Acceptance Criteria

| # | Criterion |
|---|-----------|
| AC1 | All 13 `get_latest_versions` / `transition_model_version_stage` calls replaced — zero `FutureWarning` from registry operations |
| AC2 | `apply_gate` promotes via `set_registered_model_alias("champion", new_version)`; gate-reject leaves existing `@champion` unchanged |
| AC3 | `Predictor.load_model()` resolves `models:/<name>@champion` from the MLflow artifact store (not disk). Rollback = set a different `@champion` alias; the next DAG tick serves it. Disk `model.pkl` fallback only when MLflow is unreachable. |
| AC4 | `simulator.py` and `main.py` use `get_model_version_by_alias("champion")` |
| AC5 | `NoApprovedModelError` still raised when `@champion` absent AND disk has no `model.pkl` |
| AC6 | `tests/test_container_paths.py` — 4 tests covering `container_paths` unit behaviour + train/infer path-equality (`@pytest.mark.integration`) |
| AC7 | `uv run pytest` green; `uv run mypy app tests` clean; `uv run ruff check .` clean |
| AC8 | `orchestration/README.md` updated with `mlflow.db` reset note |
| AC9 | `mlflow.lightgbm.log_model` / `load_model` replaced with `mlflow.sklearn` in `trainer.py`, `simulator.py`, `predictor.py`; `MODELS` dict accepts any `RegressorMixin`; `config.yml` `model.name: "RandomForestRegressor"` works with no further code changes |

---

## Files Changed

| File | Change |
|------|--------|
| `app/training/registry_gate.py` | Replace deprecated calls with alias API |
| `app/training/trainer.py` | Remove duplicate inline `get_latest_versions`; delegate to `apply_gate`; `mlflow.lightgbm.log_model` → `mlflow.sklearn.log_model`; `MODELS` dict typed as `dict[str, type[RegressorMixin]]`; add `RandomForestRegressor` entry |
| `app/inference/simulator.py` | Replace stages calls with alias lookup; drop Archived fallback; `mlflow.lightgbm.load_model` → `mlflow.sklearn.load_model` |
| `app/inference/predictor.py` | Replace disk-only `load_model()` with registry-primary path (`models:/<name>@champion`); `mlflow.lightgbm.load_model` → `mlflow.sklearn.load_model`; wire `tracking_uri` + `registry_model_name` from config; disk fallback preserved |
| `app/api/main.py` | Replace both `get_latest_versions` calls |
| `tests/test_container_paths.py` | New — path contract tests |
| `tests/test_train.py` | Update mocks for new alias API in `TestRegistryGate` |
| `tests/test_inference.py` | Update mocks for new fallback chain in `TestFallback` |
| `orchestration/README.md` | Add Container MLflow Setup subsection |

---

## Dependencies

- Depends on: audit PR (hardcoded-values audit, `feature/partitioned-prediction-history` merged) — **already on this branch**
- Blocks: none

---

## Risk

**Low.** The alias API is additive — existing registry entries are unaffected. Tests use
mocked `MlflowClient`, so no live registry needed. The disk fallback is a safety net for the
offline path. The only subtle risk is `MlflowException` error code handling in `Predictor.load_model()`
— catch only `MlflowException` (not bare `Exception`) so filesystem errors and
attribute errors surface immediately rather than silently falling through to disk.
