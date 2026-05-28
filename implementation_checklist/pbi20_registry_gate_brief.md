# PBI 20 — MLflow Model Registry + Champion-vs-Challenger Gate: Technical Brief

## 1. Summary

Add MLflow Model Registry integration to `app/training/trainer.py` so that every successful
training run registers a new model version and then runs a champion-vs-challenger gate: if the
challenger's test WAPE beats the current Production version's WAPE by at least
`train.gate.min_wape_lift_pct` percent (default 2.0), the challenger is promoted to Production
and the disk pickle is overwritten; otherwise it is left as a registered-but-undeployed version
and the disk pickle is untouched. The gate logic is extracted into a new
`app/training/registry_gate.py` module to keep `Trainer.run()` within the 50-line guideline.

---

## 2. Data Model / Config Changes

**`config.yml` — three new keys under the existing `mlflow:` block:**

| Key | Type | Default | Note |
|-----|------|---------|------|
| `mlflow.tracking_uri` | string | `sqlite:///mlflow.db` | Replaces current `mlruns` value |
| `mlflow.model_name` | string | `ny_taxi_demand_model` | Registry name for `mlflow.register_model` |
| `train.gate.min_wape_lift_pct` | float | `2.0` | Minimum lift percentage to pass gate |

`mlflow.experiment_name` already exists (line 99); no change needed there.

**MLflow run tags added inside `with mlflow.start_run():`**

| Tag key | Values | Condition |
|---------|--------|-----------|
| `gate_result` | `promoted` / `rejected` | Always set (AC-16) |
| `gate_rejection_reason` | non-empty string | Set only on rejection (AC-12) |

**Registered model version tag (written via `MlflowClient.set_model_version_tag`):**

| Tag key | Value | Purpose |
|---------|-------|---------|
| `model_test_wape` | float as string | Enables gate to read champion WAPE without querying run history |

No changes to Gold/Silver/Bronze parquet schemas. No new database tables.

---

## 3. Process Flow

The gate logic runs **inside** `with mlflow.start_run():`, after `mlflow.lightgbm.log_model` but
before `save_model()`. The existing run context must remain open so that `set_tag` writes to the
same run.

```
Trainer.run():
  1. [existing] train / evaluate / log params+metrics / log_model(artifact_path="model")
  2. register_model(run_uri, model_name)  → version object
  3. set_model_version_tag(model_name, version, "model_test_wape", str(model_test_wape))
     — skip tag + skip gate if wape_lift_pct is None (no baseline overlap, see AC-14/15)
  4. delegate to apply_gate(client, model_name, new_version, model_test_wape, min_lift_pct)
     which returns (gate_result: str, rejection_reason: str | None)
  5. mlflow.set_tag("gate_result", gate_result)
  6. if rejection_reason: mlflow.set_tag("gate_rejection_reason", rejection_reason)
  7. if gate_result == "promoted": save_model()   # AC-10 / AC-13
```

**`apply_gate()` decision tree (lives in `app/training/registry_gate.py`):**

```
apply_gate(client, model_name, new_version, challenger_wape, min_lift_pct):
  production_versions = client.get_latest_versions(model_name, stages=["Production"])

  if not production_versions:          # first-ever or no Production stage (AC-8, AC-15)
      transition → Production
      return ("promoted", None)

  champion = production_versions[0]
  raw_tag = champion.tags.get("model_test_wape")

  if raw_tag is None or raw_tag == "nan":   # AC: champion_wape_unavailable edge case
      return ("rejected", "champion_wape_unavailable")

  champion_wape = float(raw_tag)
  lift = compute_wape_lift(champion_wape, challenger_wape)  # reuses app.core.metrics

  if lift >= min_lift_pct:             # AC-9: >= not >
      transition new_version → Production
      transition champion → Archived
      return ("promoted", None)
  else:                                # AC-11
      return ("rejected", f"wape_lift_{lift:.2f}_below_threshold_{min_lift_pct}")
```

**Baseline-unavailable cases (no week-ago overlap, `wape_lift_pct is None`):**

- No Production incumbent → auto-promote (AC-15 satisfies AC-8).
- Production incumbent exists → reject with `gate_rejection_reason=no_baseline_for_comparison`
  (AC-14); `save_model()` skipped; disk preserved.

**Timezone / ordering:** `Trainer.run()` already uses time-based split (no shuffle). The gate
adds no temporal logic of its own, so no new timezone risk is introduced.

---

## 4. API Changes

None.

---

## 5. Frontend Changes

None.

---

## 6. Tests Required

All tests live in `tests/test_train.py::TestRegistryGate`. Mocking strategy: patch
`app.training.trainer.mlflow` (module-level, as established) **and** patch
`app.training.registry_gate.MlflowClient` to avoid real SQLite I/O. Each test is independent;
no shared state between tests.

**Happy path**

- `test_run_registers_model_version_when_run_succeeds` — `mlflow.register_model` called with
  correct run URI and `model_name` from config (AC-6, AC-7).
- `test_gate_promotes_when_no_production_version_exists` — `transition_model_version_stage`
  called with stage `"Production"`; `save_model` called once (AC-8).
- `test_gate_promotes_when_lift_meets_threshold` — challenger WAPE gives lift == min_lift_pct;
  assert stage transition to Production and prior champion to Archived (AC-9, AC-10).
- `test_gate_writes_model_test_wape_tag_on_new_version` — `set_model_version_tag` called with
  key `"model_test_wape"` and correct float string (user decision 1).
- `test_gate_result_tag_is_promoted_on_success` — `mlflow.set_tag("gate_result", "promoted")`
  called (AC-16).

**Failure cases**

- `test_gate_rejects_when_lift_below_threshold` — challenger WAPE gives lift < min_lift_pct;
  assert `transition_model_version_stage` NOT called; `save_model` NOT called (AC-11).
- `test_gate_sets_rejection_tags_when_rejected` — tags `gate_result=rejected` and
  `gate_rejection_reason` both set with non-empty value (AC-12).
- `test_gate_rejects_when_champion_wape_tag_missing` — champion version has no
  `model_test_wape` tag; assert rejection reason `champion_wape_unavailable` (edge case).
- `test_gate_rejects_with_no_baseline_when_incumbent_exists` — `wape_lift_pct=None`,
  Production version exists; assert rejection reason `no_baseline_for_comparison`; disk pickle
  unchanged (AC-14).
- `test_gate_promotes_when_no_baseline_and_no_incumbent` — `wape_lift_pct=None`, no Production
  versions; assert auto-promote (AC-15).

**Edge cases**

- `test_gate_passes_when_lift_exactly_equals_threshold` — lift == min_lift_pct; assert
  promoted (AC-9 boundary: >= not >).
- `test_trainer_reads_tracking_uri_from_config` — `mlflow.set_tracking_uri` called with value
  from config, not hardcoded string (AC-4).
- `test_gate_falls_back_to_default_model_name_when_key_missing` — config dict has no
  `mlflow.model_name`; `register_model` called with `"ny_taxi_demand_model"` (AC-2 / edge case).
- `test_gate_falls_back_to_default_min_lift_when_key_missing` — config dict has no
  `train.gate`; gate threshold defaults to `2.0` (AC-3 / edge case).

---

## 7. Risks and Open Questions

- **MLflow stage string API stability.** Stage names `"Production"` and `"Archived"` still work
  in MLflow 3.11.1 via backward-compatibility shim (confirmed by researcher). If MLflow removes
  stage support in a future minor version, PBI 21 will need to migrate to aliases. Track as a
  dependency risk; no action needed now.
- **`model_test_wape` is None at gate time.** When `wape_lift_pct is None` and an incumbent
  exists, the spec calls for rejection. However the registered version will have no
  `model_test_wape` tag, making it appear to future runs as if the champion WAPE is unavailable.
  This is acceptable: a version with no WAPE tag will never be promoted over a valid champion,
  which is the safe behavior. Document in the tag-writing logic.
- **Concurrent training runs.** If two runs register simultaneously, both may see the same
  Production version and both attempt to promote. SQLite provides no row-level locking for
  MLflow registry operations. This is unlikely in the current single-operator setup but should
  be noted for any future CI parallelism.
- **`compute_wape_lift` in `registry_gate.py` imports `app.core.metrics`.** Confirm PBI 18
  is merged before this branch is cut; otherwise the import will fail at test time.
- **`mlflow.db` write path.** `sqlite:///mlflow.db` resolves relative to the process CWD, not
  the project root, unless invoked from the project root. This matches the existing convention
  but should be noted in the README.

---

## 8. Files That Will Change

**Backend**

- `config.yml` — add `mlflow.model_name`, `mlflow.tracking_uri` (sqlite), `train.gate.min_wape_lift_pct`
- `app/training/trainer.py` — call `set_tracking_uri` from `__init__`; add registry + gate
  delegation inside `run()` (inside the existing `with mlflow.start_run():` block); conditionally
  call `save_model()`; read `mlflow.model_name` and `train.gate.min_wape_lift_pct` from config
- `(new) app/training/registry_gate.py` — `apply_gate()` function; keeps `Trainer.run()` <50 lines

**Tests**

- `tests/test_train.py` — add `TestRegistryGate` class with all tests listed in §6

**Config / Docs**

- `implementation_checklist/pbi20_registry_gate_brief.md` — this document (already written)
