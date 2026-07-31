# Test Suite Housecleaning Report

_Date: 2026-07-31 · Branch: `feature/pbi-25c-train-monthly-dag`_

> **STATUS: EXECUTED** — Bucket A deleted; Bucket B handled (re-point + rewrite). See
> [Actions taken](#actions-taken-executed) at the bottom. The analysis below is the
> investigation record that drove those actions.

## Summary

- **36 test files · ~476 collected tests.**
- The suite **currently cannot run clean**: `tests/test_exp04_predict.py` fails collection (`ModuleNotFoundError: orchestration_experiment.dags.exp04_predict`).
- Method: for each suspect file, checked (a) does the imported subject exist on disk, (b) does the test actually run/pass, (c) production vs experiment scaffold, (d) duplicated coverage. **Running the tests — not just reading imports — was decisive.**

## Key finding: two parallel copies of six helpers

The pipeline helpers exist twice, with **verified-identical public API** (`diff` = SAME):

| Experiment (scaffold) | Production (promoted) |
|---|---|
| `orchestration_experiment/dags/exp0*.py` | `orchestration/lib/*_helpers.py` |

The promotion moved the **code** to `orchestration/lib/` but the **tests were never re-pointed**. Consequence:

| Concern | State |
|---|---|
| Production `orchestration/lib/` helpers | **Zero direct test coverage** |
| Experiment `orchestration_experiment/dags/exp0*` helpers | Covered **twice** (`test_exp0*` **and** `test_*_helpers`) |

---

## Bucket A — Dead, delete now (1 file)

| File | Evidence |
|---|---|
| `tests/test_exp04_predict.py` | Imports `orchestration_experiment.dags.exp04_predict` → **not on disk, never committed**. **Breaks `pytest` collection for the whole suite.** Superseded by `orchestration/lib/predict_helpers.py` + `log_predict_helpers.py`, which have their own passing tests. |

**Recommendation:** delete — guards nothing, actively breaks `pytest`.

---

## Bucket B — Redundant duplicates (6 files, ~56 tests)

Each pair tests the **same experiment module**. `test_exp0*` were the incremental-build scope-guards; `test_*_helpers` are the "real" unit tests — but **both currently point at the experiment copy**.

| Experiment scope-guard (candidate to drop) | "Real" test (keep, but re-point) | Subject |
|---|---|---|
| `test_exp02_ingest.py` | `test_ingest_helpers.py` | ingest |
| `test_exp03a_validate.py` | `test_validate_helpers.py` | validate |
| `test_exp03b_feature.py` | `test_feature_helpers.py` | feature |

Memory `exp-dag-promotion-notes` supports dropping the `test_exp0*` scope-guards **after promotion** — which has happened. **But** only re-point the survivors first; otherwise deleting the EXP tests leaves production untested.

---

## Bucket C — Keep

All 25 remaining files test live `app/` or current-PBI code and pass.

**Explicitly NOT dead** (earlier misflagged): `test_predict_helpers.py`, `test_log_predict_helpers.py`, `test_boundary_helpers.py` — **10 passing**, subjects live in `orchestration/lib/` (resolved via `conftest` `sys.path`).

---

## Recommendation

1. **Now:** delete `tests/test_exp04_predict.py` (unblocks the suite).
2. **Proper fix (small, high-value):** re-point `test_ingest_helpers.py` / `test_validate_helpers.py` / `test_feature_helpers.py` imports from `orchestration_experiment.dags.exp0*` → `orchestration.lib.*_helpers`, then delete the `test_exp02/03a/03b` duplicates. Net: production gets real coverage; ~34 redundant tests removed; experiment scaffold can later be deleted cleanly.
3. **Do not** blind-delete the EXP tests without step 2 — that leaves `orchestration/lib/` with no tests.

> The real issue isn't "too many tests" — it's that **production helpers are untested while the passing tests guard a copy you intend to discard.**

---

## Actions taken (EXECUTED)

### Bucket A — deleted
- `tests/test_exp04_predict.py` — removed (dead import, was breaking `pytest` collection).

### Bucket B — re-pointed survivors, deleted duplicates
Deleted the experiment scope-guard duplicates:
- `tests/test_exp02_ingest.py`
- `tests/test_exp03a_validate.py`
- `tests/test_exp03b_feature.py`

Re-pointed the survivor tests (imports **and** `patch()` targets) from
`orchestration_experiment.dags.exp0*` → `orchestration.lib.*_helpers`:
- `tests/test_ingest_helpers.py` — re-pointed, passes as-is (API identical).
- `tests/test_validate_helpers.py` — re-pointed, passes as-is (API identical).
- `tests/test_feature_helpers.py` — **rewritten**: the production API had diverged
  from the experiment copy, so a plain re-point failed. New tests cover the
  production contract:
  - `build_feature_config` no longer takes `window_end` (OQ4 cutoff removed).
  - `run_feature` is **transform-only**: reads pre-fitted IQR bounds from
    `model.store_path/iqr_bounds.json`, calls `clean(raw, iqr_bounds=...)`,
    saves silver, then runs `FeatureEngineer`.
  - **Anti-regression test**: asserts `DataCleaner.run()` is *never* called
    (guards against reverting to the re-fitting path the rewrite eliminated).

### Verification
- Full suite: **435 passed, 15 skipped** (previously could not collect).
- Touched files: `ruff check` ✅ · `ruff format` ✅ · `mypy` ✅.
- One remaining `mypy` error at
  `orchestration/dags/the_15_minute_refresh_inference_dag.py:171` is
  **pre-existing** (confirmed via `git stash`) and outside this changeset.

### Net effect
- 4 redundant/dead test files removed.
- Production `orchestration/lib/{ingest,validate,feature}_helpers.py` now have
  direct unit coverage — previously none (tests guarded the experiment copies).
- The `orchestration_experiment/` scaffold can now be deleted cleanly whenever
  the orchestrator redesign lands, with no test coverage lost.
