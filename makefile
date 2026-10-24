# NYC Taxi Demand Forecasting — Makefile
#
# One-liner ergonomics for the two things a reviewer actually types:
#   1. Quality gates before pushing (make check)
#   2. The mode-2 demo path (make airflow-up / make demo / make demo-tick)
#
# Everything else here is a thin wrapper over `uv run …` or `docker compose …`.

# PYTHONPATH must include the repo root (for app/) and orchestration/lib (for
# *_helpers modules). Set once here so all fast_forward targets inherit it.
FAST_FORWARD_ENV := PYTHONPATH=$(CURDIR):$(CURDIR)/orchestration/lib

# --- Dev environment ---------------------------------------------------------

setup:
	uv sync

# --- Quality gates (mirror scripts/ci-*.sh) ----------------------------------

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy app tests

test:
	uv run pytest

# Aggregate: runs the same four gates CI enforces. Pre-push shortcut.
check: lint format-check typecheck test

# --- Orchestration (Airflow container stack) ---------------------------------
# Bootstrap once: `cp orchestration/.env.example orchestration/.env`
# then `make airflow-init`. See orchestration/README.md.

COMPOSE_FILES = -f orchestration/docker-compose.yaml

airflow-init:
	docker compose $(COMPOSE_FILES) up airflow-init

airflow-up:
	docker compose $(COMPOSE_FILES) up -d

airflow-down:
	docker compose $(COMPOSE_FILES) down

# --- Demo --------------------------------------------------------------------
# Two entrypoints:
#   make demo       -- FastAPI serving cached predictions on :8000
#   make demo-tick  -- Fire one DAG run to advance the sim watermark 15 min
#
# Live cadence: `make airflow-up`, then repeat `make demo-tick` to walk
# the 2021 timeline. To replay from seed use `make reset-watermark` first.
# Teardown: `make demo-stop`.

demo:
	@echo "[demo] Static demo -- serving cached predictions on :8000"
	uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000

demo-tick:
	@echo "[demo-tick] Firing one 15_minute_refresh_inference_dag run"
	docker compose $(COMPOSE_FILES) exec -T airflow-scheduler airflow dags trigger 15_minute_refresh_inference_dag

demo-populate:  ## one-shot bronze→gold for Q1 2021 + set watermark (run before training)
	$(FAST_FORWARD_ENV) uv run python scripts/bulk_advance.py --until 2021-03-31

demo-jump-to-jan8:  ## bulk-advance to 2022-01-08 so DAG can start live from there
	$(FAST_FORWARD_ENV) uv run python scripts/bulk_advance.py --until 2022-01-08

demo-advance:  ## advance inference watermark → 2022-01-31 (15-min ticks through Jan 2022, needs model)
	$(FAST_FORWARD_ENV) uv run python scripts/fast_forward.py --until 2022-01-31

demo-advance-full:  ## advance inference watermark → 2022-03-31 (full Q1 2022 cadence: 3 monthly retrains)
	$(FAST_FORWARD_ENV) uv run python scripts/fast_forward.py --until 2022-03-31

demo-stop:
	docker compose $(COMPOSE_FILES) down

# --- Reset -------------------------------------------------------------------
# Kept separate on purpose. reset-mlflow nukes registry metadata baked with
# a host-absolute artifact_location -- opt in deliberately, not as a side
# effect of a generic `reset`.

reset-watermark:
	rm -f data/watermarks/refresh_15min/.current_watermark

reset-mlflow:
	rm -f data/mlflow.db

# --- Clean -------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache

.PHONY: setup lint format format-check typecheck test check \
        airflow-init airflow-up airflow-down \
        demo demo-tick demo-populate demo-jump-to-jan8 demo-advance demo-advance-full demo-stop \
        reset-watermark reset-mlflow clean
