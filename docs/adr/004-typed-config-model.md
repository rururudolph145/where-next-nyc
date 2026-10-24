# ADR 004 — Typed Pydantic config model

- **Status:** Proposed
- **Date:** 2026-08-10
- **Context:** Follow-up to the hardcoded-values audit (ADR forward-referenced in `docs/superpowers/plans/2026-08-08-hardcoded-values-audit.md`)
- **Related:** `docs/superpowers/specs/2026-08-10-typed-config-model-design.md`

---

## Context

`app/core/config.load_config()` returns `dict[str, Any]`. Every consumer
accesses values via `config["section"]["key"]` — a missing key or wrong type
silently survives until that code path is exercised at runtime. Several
consumers also carry `.get("key", default)` fallbacks whose defaults duplicate
values already present in `config.yml`; after the hardcoded-values audit these
fallbacks are dead code.

This is a behavior-changing fix: failure moves from **use-time** (`KeyError` /
wrong type silently coerced) to **load-time** (`pydantic.ValidationError`).
That is why it was kept out of the behavior-preserving audit PR.

---

## Decision

Replace the raw-dict return value of `load_config()` with a hierarchy of
frozen Pydantic v2 `BaseModel` sub-classes (`AppConfig`). All consumer class
signatures accept `AppConfig` instead of `dict[str, Any]`. The
`orchestration/lib/build_*_config()` override helpers use `model_copy` instead
of `deepcopy` + dict mutation.

---

## Consequences

### Positive

- **Fail-fast:** a misconfigured `config.yml` raises `ValidationError` at
  startup, not mid-pipeline.
- **Typed access:** `config.clean.iqr_multiplier` is a `float` by
  construction; mypy flags any remaining subscript access after migration.
- **Dead code removed:** all `.get(key, default)` fallbacks that duplicate
  `config.yml` values are deleted.
- **Immutability explicit:** `frozen=True` on `AppConfig` replaces the
  implicit `deepcopy` convention in the override helpers — the only
  mutation path is `model_copy`, which is auditable.

### Negative / risks

- **Test fixture churn:** tests that pass raw `dict` as `config=` to class
  constructors need mechanical updates to `AppConfig.model_validate({...})`.
  Blast radius is contained by doing this as the final migration step.
- **`model_copy` verbosity:** nested overrides require chaining
  `model_copy` calls. Mitigated by keeping the `build_*_config()` helpers
  as the canonical override seam — the verbosity is localised there.
- **`pydantic` becomes an explicit dependency:** it is already installed
  (v2.13.2) but not declared in `pyproject.toml`. Must be added.

### Neutral

- `ingest.bronze_schema`, datetime strings (`dataset_max`, `seed_start`),
  and `model.params` remain loosely typed (`dict[str, str]`, `str`,
  `dict[str, Any]`). Stricter validators are a natural follow-up, out of
  scope here.
