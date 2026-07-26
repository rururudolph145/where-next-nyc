python = .venv/bin/python
pip = .venv/bin/pip

setup:
	python3 -m venv .venv
	$(.venv/bin/python) -m pip install --upgrade pip
	$(pip) install -r requirements.txt

run:
	$(python) -m pipeline.run_all

mlflow:
	.venv/bin/mlflow ui

test:
	$(python) -m pytest

clean:
	rm -rf steps/__pycache__
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf tests/__pycache__

remove:
	rm -rf .venv
	rm -rf mlruns

# --- Orchestration (PBI 25a) -------------------------------------------------
# Container-only Airflow. Bootstrap with `cp orchestration/.env.example orchestration/.env`
# before running `make airflow-init`. See orchestration/README.md.
#
COMPOSE_FILES = -f orchestration/docker-compose.yaml

airflow-init:
	docker compose $(COMPOSE_FILES) up airflow-init

airflow-up:
	docker compose $(COMPOSE_FILES) up -d

airflow-down:
	docker compose $(COMPOSE_FILES) down

# --- Demo (PBI 25b) -----------------------------------------------------------
# Two demo entrypoints:
#   make demo       -- Static demo. Starts FastAPI serving cached predictions.
#   make demo-tick  -- Fire one DAG run against a running Airflow stack.
#
# Live cadence: `make airflow-up` then repeat `make demo-tick` to walk
# the 2021 timeline one 15-minute slice per trigger.
#
# Teardown: make demo-stop  (stops FastAPI and the Airflow stack).

demo:
	@echo "[demo] Static demo -- serving cached predictions"
	uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000

demo-tick:
	@echo "[demo-tick] Firing one 15_minute_refresh_inference_dag run"
	docker compose $(COMPOSE_FILES) exec -T airflow-scheduler airflow dags trigger 15_minute_refresh_inference_dag

demo-stop:
	docker compose $(COMPOSE_FILES) down

.PHONY: setup run mlflow test clean remove airflow-init airflow-up airflow-down demo demo-tick demo-stop
