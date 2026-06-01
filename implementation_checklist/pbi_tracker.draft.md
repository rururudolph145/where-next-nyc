# PBI Tracker — NYC Taxi Demand Forecasting

> **Restructured 2026-05-31.** This tracker is organised around **six demo modes** — each independently demoable, each with its own reviewer-facing story. The previous activity-and-phase nesting is preserved as legacy notes inside each demo-mode block, but the top-level question this tracker now answers is: *"what is the minimum work to demo X?"*
>
> Demo modes intentionally avoid letter ordering (`L0..L5`) to prevent confusion with Google's engineering ladder. There is no implied "more is better" — fewer modes, each polished, beats more modes half-finished. A reviewer can stop at any mode and have a coherent story.

---

## Demo Mode Index

| Mode | Reviewer test | Status |
|---|---|---|
| **Static demo** | `make demo` → curl `/heatmap` → ranked predictions + baseline lift | ✅ shipped |
| **Live cadence** | `DEMO_MODE=1 make demo` → heatmap refreshes every ~2s = 15 sim min | 🟡 next (PBI 25b) |
| **Closed training loop** | Trigger retrain → gate decision visible → register or roll back | 🔵 scoped (PBI 25c — drop; document) |
| **Closed feedback loop** | Inject drift → alert → auto-retrain | 🔵 scoped (PBIs 23/24/25d — drop; document) |
| **Outcome harness** | Per-policy trips-served lift in `docs/simulation_results.md` | 🔵 scoped (PBI 28 — high-value optional build) |
| **Personalised ranking** | `/recommend?lat=&lon=` → top-N zones by demand/cost | ⚪ deferred (PBIs 5–8) |

---

# Demo Mode 1 — Static Demo (✅ DEMOABLE TODAY)

> **Reviewer test:** `uv run` → `curl /heatmap` → ranked predictions with baseline lift, `/health`, `/model-info`.
>
> The driver-facing question — *"where will demand be 30 minutes from now"* — is answered end-to-end on a static historical dataset. Bronze → Silver → Gold pipeline, LightGBM validated against a last-week-same-hour baseline, FastAPI heatmap with fallback to the last approved model.

### Forecasting core (Activity 1–3)

- [x] **PBI 1 — Single-Horizon Forecasting Engine (time t)** — *shipped*
- [x] **PBI 2 — Inference Pipeline (time t)** — *shipped*
- [x] **PBI 3 — Heatmap API (`GET /heatmap`)** — *shipped*

### Nowcast B-track (15-min bucket + spatial smoothing + 30-min summed target)

- [x] **B0 — Synthetic-Data E2E Test Suite** — *shipped*
- [x] **B1 — 30-Min Bucket + Neighbour-Mean Lag Smoothing** — *shipped*
- [x] **B2 — 15-Min Bucket + 30-Min Summed-Horizon Target** — *shipped*

### Ingestion performance hardening

- [x] **PBI 12 — Lock-in Bronze Temporal-Sort Invariant (Test-First)** — *shipped in `b3ba4d9`*
- [x] **PBI 13 — Parallel-Equivalence Regression Test** — *shipped*
- [x] **PBI 14 — Parallelise `load_raw` with `ThreadPoolExecutor`** — *shipped*

### Credibility layer — turns the README's value claim into a number

- [x] **PBI 18 — Baseline Model + Recall@K + WAPE-Lift Reporting** — *shipped in `783f5ff`*
- [x] **PBI 19 — Pandera Schema Gate Between Bronze and Silver** — *shipped in `d317be0`*
- [x] **PBI 27 — `/health` + `/model-info` + Simulated-Clock Flag** — *shipped on `refactor/app-package-layout`*

### Safety layer — registry gate + inference fallback

- [x] **PBI 20 — MLflow Model Registry + Champion-vs-Challenger Gate** — *shipped in `17960de`*
- [x] **PBI 21 — Inference Fallback to Previous Approved Model** — *shipped in `ac0e73c`*

### Monitoring foundation (data-foundation only; detector is dropped)

- [x] **PBI 22 — Prediction Log + Actuals-Join Job** — *shipped in `6d6854f`*

### CI/CD gate — every change from here lands behind enforced gates

- [x] **PBI 30 — GitHub Actions CI/CD: Lint + Test** — *shipped in `98761fd` (PR #13 → `dev_MVP`)*

### Multi-horizon shift (deferred — A3 knob already lets us flip)

- [ ] **PBI 4 — Multi-Horizon Forecasting (t+30, t+60)** — *deferred; superseded for MVP by B-track; `add_shifted_target` knob already ships (A3, commit `4ea0a03`)*

---

# Demo Mode 2 — Live Cadence (🟡 NEXT — ONE PBI AWAY)

> **Reviewer test:** `DEMO_MODE=1 make demo` → heatmap visibly refreshes every ~2 seconds of wall-clock as 15 simulated minutes elapse. Midtown lights up at simulated 5pm; JFK at simulated 6am.
>
> Adds no ML capability — adds **operational realism.** This is the version of the demo that proves the system is operationally coherent under time. The full implementation plan lives in `pbi25b_refresh_15min_dag_brief.md`.

### Orchestrator foundation (shipped)

- [x] **PBI 25a — Orchestrator Foundation (`Clock` abstraction + `orchestration/` skeleton)** — *shipped in `ff466e9`*
  - 29 tests added (13 unit + 16 acceptance)
  - One-way dependency invariant `app/` ↛ `orchestration/` enforced at AST level
  - Existing `makefile` targets preserved alongside three new `airflow-*` targets

### The single PBI that unlocks Mode 2

- [ ] **PBI 25b — `refresh_15min_dag` (Ingestion → Inference Cadence)**
  - Add `orchestration/dags/refresh_15min_dag.py` — `ingest → validate (PBI 19) → feature → predict → log (PBI 22)`
  - Driven by the injected `Clock` from PBI 25a, never `datetime.utcnow()`
  - Add `DEMO_MODE=1` factory selecting `WallClock` vs `SimulatedClock(speedup=450)`
  - Add `make demo` target wired to bring up FastAPI + Airflow scheduler under the factory
  - Add `tests/test_clock_discipline.py` repo-wide grep — no production module imports `datetime.utcnow` or `time.sleep` directly
  - Acceptance:
    - `DEMO_MODE=1 make demo` brings up a running stack; `curl /heatmap` shows `bucket_time` advancing by 15 sim-min per ~2s wall-clock
    - `/health` reports `{"simulated": true, "simulated_now": "...", "wall_clock_now": "..."}` when `DEMO_MODE=1`
    - DAG idempotent on the same `(date, bucket)` watermark (rerunning the same bucket is a no-op)
    - All quality gates green (`ruff`, `mypy`, `pytest`)
  - Depends on: PBI 25a (foundation), PBI 19 (validate step), PBI 22 (log step)
  - **This is the highest-leverage open PBI in the tracker.**

---

# Demo Mode 3 — Closed Training Loop (🔵 SCOPED — DROP, DOCUMENT)

> **Reviewer test:** trigger manual training → PBI 20 gate decision visible → if rejected, the previous model continues serving (visible in `/model-info`).
>
> The gate (PBI 20) and fallback (PBI 21) **already work as library calls** — the static demo exercises them. Wrapping them in Airflow proves no new capability. **Drop the DAG; document the wrapper in the README's "Future Work" section.**

- [ ] ~~**PBI 25c — `train_monthly_dag`**~~ — *dropped from build queue; documented in README "Future Work"*
  - Original scope: `orchestration/dags/train_monthly_dag.py` — cron `@monthly` + manual trigger; train → gate (PBI 20) → register
  - Reason for drop: capability exists as library code; wrapping in Airflow adds operational polish, not ML capability. Build cost > interview signal.

---

# Demo Mode 4 — Closed Feedback Loop (🔵 SCOPED — DROP, DOCUMENT)

> **Reviewer test:** inject synthetic drift → detector flags it → alert writes `retrain_required` marker → monitor DAG triggers training DAG.
>
> Drift monitoring on a **finite historical dataset is largely theatre** — there is no real drift to detect on TLC parquets you've already seen. The prediction log (PBI 22) is the data foundation, which is enough to *talk about* closed-loop monitoring in interviews. Do not build the detector; describe the design in "Future Work".

- [ ] ~~**PBI 23 — Drift & Performance Detector (Evidently)**~~ — *dropped; documented*
  - Original scope: `app/monitoring/drift.py` wrapping Evidently `DataDriftPreset` + `RegressionPreset` over a rolling 7-day window from PBI 22's joined frame
  - Reason for drop: drift on static historical data is theatre; design itself is sufficient interview signal

- [ ] ~~**PBI 24 — Alerts + Retrain Trigger**~~ — *dropped; documented*
  - Original scope: `app/monitoring/alert.py` reading PBI 23's report; emits `retrain_required` marker
  - Reason for drop: depends on PBI 23 (dropped)

- [ ] ~~**PBI 25d — `monitor_hourly_dag`**~~ — *dropped; documented*
  - Original scope: hourly DAG running PBI 23 detector; `FileSensor` on PBI 24's marker fires `train_monthly_dag`
  - Reason for drop: depends on 23/24 (both dropped)

---

# Demo Mode 5 — Outcome Harness (🔵 OPTIONAL — HIGHEST-VALUE POST-MODE-2 BUILD)

> **Reviewer test:** open `docs/simulation_results.md` → per-policy trips served / % idle / miles to pickup / total fare on a held-out week.
>
> This is the only piece that converts "the model has lift on WAPE" into "drivers using it earn more dollars." If portfolio time allows exactly one more PBI after Mode 2 ships, this is the one — the slide that wins SmartNews / Google interview conversations.

- [ ] **PBI 28 — Driver-Agent Simulation Harness**
  - Add `app/simulation/` package with three policies:
    - `HeatmapFollower` — calls `/heatmap`, drives to a top-K predicted zone
    - `LastWeekFollower` — baseline; reads PBI 18's last-week-same-hour lookup, picks top-K from *that*
    - `Stationary` — control; never moves
  - At each simulated 15-min tick (driven by the same `SimulatedClock` from PBI 25a):
    - Idle drivers pick a destination per their policy
    - En-route drivers travel
    - On arrival, the harness consults the historical trip table; if a real trip started in that zone during the window and a driver is there, the driver gets the trip (locked into its historical duration)
  - Output: per-policy table in `docs/simulation_results.md`
  - Acceptance:
    - All three policies run on a held-out week without crashing
    - The output table is populated with finite numbers
    - HeatmapFollower's lift over LastWeekFollower is reported honestly — if zero or negative, the README says so
  - Depends on: PBI 18 (baseline policy), PBI 25a (SimulatedClock) — both shipped

---

# Demo Mode 6 — Personalised Ranking (⚪ DEFERRED — README ONLY)

> **Reviewer test:** `GET /recommend?lat=&lon=` → top-N zones ranked by predicted demand divided by travel cost.
>
> Promotes the heatmap (Uber model) to ranked per-driver recommendations (DiDi model). Documented as design trade-off in the README; not built. *Documenting the trade-off — heatmap leaves interpretation to the driver, ranking centralises it but requires fleet-state tracking to avoid herding — is itself interview signal.*

- [ ] **PBI 5 — Zone Mapping Utility** — *deferred*
- [ ] **PBI 6 — User Location Capture** — *deferred*
- [ ] **PBI 7 — Distance-Decay Scoring** — *deferred*
- [ ] **PBI 8 — Top-N Recommendation API** — *deferred*

---

# Cross-Cutting — API & Ingestion Hardening

Independent of demo modes; pick when relevant.

- [ ] **PBI 16 — Paginate `GET /heatmap` (`?top_n=`)** — partial
- [ ] **PBI 29 — Latency SLO Baseline + CI Guard** — depends on PBI 30 (shipped)
- [ ] **PBI 15 — Streaming Bronze Write (Generator Compliance)** — stretch; defer until ingestion OOMs in practice
- [ ] **PBI 17 — Bring Socrata Downloader Under Performance Rules** — blocked; requires downloader migration into `app/`
- [ ] **PBI 26 — In-Memory Online Feature Store** — stretch; depends on PBI 25b; defer until on-demand inference is required

### Post-MVP reliability

- [ ] **PBI 9 — Stale Data Handler** — historical-averages fallback; complementary to PBI 21's model-level fallback
- [ ] **PBI 10 — Model Monitoring (L5 signal)** — superseded by PBI 22 + the dropped 23/24
- [ ] **PBI 11 — Near-Real-Time Inference with Simulated Clock** — largely subsumed by PBI 25a's `SimulatedClock`

---

# Priority Order — What To Do Next

1. **Ship PBI 25b** (Live Cadence) — single highest-leverage open PBI. See `pbi25b_refresh_15min_dag_brief.md`.
2. **Polish the README** — rename `README.draft.md` → `README.md` once reviewed. Embed `sample_heatmap.png`. Fill the baseline-lift table from PBI 18's reporting step.
3. **Walk-forward CV** for the baseline-vs-LGBM comparison — one afternoon's work; large credibility bump.
4. **Per-zone error analysis notebook** — one committed notebook showing where the model fails (airport zones at midnight, sparse zones, special events).
5. **Write "Future Work" sections** covering Modes 3, 4, 6 — design paragraphs, not code.
6. *(Optional, only if time remains.)* **Ship PBI 28** (Outcome Harness, Mode 5).

After step 6, **stop.** Record a 60-second Loom of `DEMO_MODE=1 make demo`. Ship the portfolio.

---

# Dependency Graph

```
Static demo (Mode 1, shipped)
  PBI 1 → PBI 2 → PBI 3
  B0 → B1 → B2
  PBI 12 → PBI 13 → PBI 14
  PBI 18 → PBI 20 → PBI 21
  PBI 19, PBI 22, PBI 27, PBI 30

Live cadence (Mode 2, next)
  PBI 25a (shipped) → PBI 25b ★ ← THIS UNLOCKS MODE 2

Outcome harness (Mode 5, optional high-value)
  PBI 18 (shipped) + PBI 25a (shipped) → PBI 28

Cross-cutting
  PBI 30 (shipped) → PBI 29
  PBI 25b → PBI 26 (stretch, only if on-demand inference needed)

Dropped / documented in README
  PBI 25c (Closed training loop DAG)
  PBI 23 → PBI 24 → PBI 25d (Closed feedback loop)

Deferred / README only
  PBI 5, 6, 7, 8 (Personalised ranking)
  PBI 9, 10, 11
```

---

# Known Limitations & Future Considerations

- **Multi-driver coordination.** MVP serves the same heatmap to all drivers (Uber model). DiDi-style per-driver optimisation requires fleet-state tracking and is deferred to post-MVP.
- **No live data.** TLC publishes on a 1–2 month lag; "live" production is honestly impossible from a portfolio repo. The compressed `SimulatedClock` exercises the same code path production would.
- **Drift monitoring is theatre on a static dataset.** The prediction log (PBI 22) is the foundation; design is documented in README "Future Work".
- **Macro patterns only.** Weather, events, transit disruptions not yet incorporated.

---

# Legacy Tracker Notes (preserved for reference)

The pre-restructure tracker organised work as: *Release: MVP* → *Performance Hardening* → *Architecture Alignment* → *Closed-Loop Monitoring* → *Production Orchestrator* → *Post-MVP*. That framing is preserved in git history (file: `pbi_tracker.md` pre-2026-05-31). The demo-mode restructure above is a strict reorganisation — no PBIs were renumbered, no acceptance criteria changed; only the grouping and the dropped/scoped designations are new.
