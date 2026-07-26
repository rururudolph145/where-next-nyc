# ADR 001 — Bronze storage: single-file Parquet (full-refresh)

> The Bronze layer is written as a single, overwritten Parquet file rebuilt
> from the full raw range on every run. This is a deliberate choice for a
> fixed-window, single-writer portfolio demo — not an oversight. This record
> captures the reasoning, the latent limitations, and the trigger that would
> justify migrating to a partitioned + compacted layout.

- **Status:** Accepted
- **Date:** 2026-06-10
- **Context:** PBI 25b (`refresh_15min_dag`, 15-minute simulated refresh cadence)
- **Related:** `docs/system_design.md` (medallion architecture, "single-node,
  local-first" non-goals), `orchestration/dags/refresh_15min_dag.py`
  (watermark idempotency)

---

## Context

The `refresh_15min_dag` runs the medallion pipeline
(`raw → bronze → silver → gold → predictions`) on a 15-minute cadence. Each
run rebuilds Bronze via `Ingestion.run()`:

1. `load_raw()` reads **every** daily parquet in the config-pinned range
   (`config.yml: ingest.start_year_month → end_year_month`, currently
   `2021-01 → 2021-03`) and `pd.concat`s them into one in-memory DataFrame.
2. `save_bronze()` writes that whole DataFrame to a **single** file,
   `data/bronze/yellow_tripdata/bronze.parquet`, via `df.to_parquet(path)`.

`pandas.to_parquet()` to a file path truncates and rewrites — there is no
append mode. So every run is a **full refresh**: the entire Bronze file is
recomputed from the entire raw range and overwritten.

The 15-minute cadence is **simulated** over a **fixed** historical window.
`raw/` does not accumulate in this demo; idempotency across re-runs is provided
by a watermark sentinel (`data/watermarks/refresh_15min/{date}/{bucket}.done`),
which causes a re-run of an already-completed `(date, bucket)` to skip rather
than reprocess.

## Decision

**Keep Bronze as a single, full-refresh Parquet file. Change nothing in the
storage layer for the portfolio demo.**

This is the YAGNI-correct choice for the workload as it actually exists:

- **Fixed window** — `raw/` does not grow, so the per-run cost is bounded and
  constant, not unbounded.
- **Single writer** — the DAG uses `max_active_runs=1`; there is no concurrent
  write or read-during-compaction hazard.
- **Append-only facts** — taxi trips are immutable; there are no updates or
  deletes that would need upsert/merge semantics.
- **Full-refresh is the *correct* shape for a batch backfill over a fixed
  range** — which is exactly what this pipeline was built for.

## Format vs. layout — the key distinction

The limitations below are **layout/lifecycle** concerns, not **file-format**
concerns. Parquet remains the right *format* at every rung of the ladder; the
only open question is how many files there are, how they are named, and how
their visibility is coordinated.

- **Switching file format does not help.** Row formats (Avro/CSV/JSON) are
  appendable but discard the columnar compression, column projection, and
  predicate pushdown that the `feature` and serving reads depend on. ORC is a
  lateral move (same small-files behaviour, weaker Python ecosystem).
- **The mature answer keeps Parquet and changes the *layout*** — partitioned
  directories, optionally with a transaction log. Iceberg, Delta Lake, and
  Hudi all store their data **as Parquet files**; they add a metadata layer on
  top, they do not replace the format.

## Consequences

### Accepted (latent for a fixed window, active if `raw/` ever grows)

- **O(history) cost per run.** A streaming/micro-batch cadence should do
  O(new data) work per run; full-refresh does O(all data). At the fixed 2021
  window this is a bounded constant. At production scale (years of growing
  raw) it becomes a multi-GB `pd.concat` every 15 minutes — an out-of-memory
  ceiling, not merely slow.
- **Non-atomic whole-file rewrite.** A run that dies mid-write leaves a
  truncated Bronze with no prior version (Bronze is gitignored — no
  `git restore`). A partitioned layout that only ever *adds* files would lose
  at most the in-flight slice.

### Not incurred (because we did **not** partition)

- No small-files explosion. (A naive per-bucket layout would produce ~96
  files/day ≈ 35k files/year, with the attendant metadata-read, compression,
  and listing costs — see the migration path below.)

## Alternatives considered

| Option | Verdict |
|--------|---------|
| Replace Parquet with Avro/CSV/JSON (appendable) | **No** — strict downgrade; loses columnar performance to solve a problem the layout solves better. |
| Replace Parquet with ORC (columnar) | **No** — lateral move, weaker Python support, identical small-files behaviour. |
| Partition Bronze by `(dt, bucket)`, append new partition files | **Deferred** — the correct rung-2 step; unjustified while `raw/` is fixed (YAGNI). |
| Partition + scheduled compaction (rung 3a) | **Deferred** — introduces a compaction/read race the demo never triggers. |
| Adopt a table format (Iceberg / Delta / Hudi) over Parquet | **Iceberg-if-ever** — Iceberg fits the shape best (engine-agnostic, PyIceberg, append-only, single-writer). Over-engineering for a fixed-window demo; valuable as a *documented* decision, not as code. |
| DuckDB as a query layer over Parquet | **Optional ergonomics** — gives "query a pile of Parquet as one table" without a table format; a read/query convenience, not a storage change. |

## Migration trigger

Revisit this decision **when `raw/` becomes a live or growing source** (i.e.
the cadence stops being simulated over a fixed window). The migration path,
in order of escalating need:

1. **Partition Bronze** by `(dt, bucket)`:
   `data/bronze/yellow_tripdata/dt=YYYY-MM-DD/bucket=NN/part.parquet`. `ingest`
   reads only the new raw slice and writes a new partition file; downstream
   readers glob the directory (`pd.read_parquet(dir)`) and stay unchanged so
   long as the schema contract holds. Cost drops to O(new); writes become
   crash-safe (new file, never mutate old).
2. **Add daily compaction** once partition count is a concern: a separate,
   slower job that rolls a sealed day's micro-files into one well-sized
   (~128MB–1GB) daily file. The existing `(date, bucket)` watermark is the
   natural completeness oracle — compact a day only when all 96 buckets are
   `.done`, so compaction never races the writer.
3. **Adopt a table format (Iceberg via PyIceberg)** when serving and
   compaction can run **concurrently**. At that point a directory glob has a
   read-consistency hazard (a reader can see both a compacted file and its
   not-yet-deleted staging sources → double counting) that no schema contract
   can fix. A transaction/manifest log provides atomic snapshot visibility —
   readers ask the log, not the directory. This is the correctness boundary
   that makes a table format necessary rather than merely nice.

Each rung solves the previous rung's problem and exposes the next; the demo
correctly sits at rung 1 because every later problem is latent under a fixed
window with a single writer.
