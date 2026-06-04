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

airflow-init:
	docker compose -f orchestration/docker-compose.yaml up airflow-init

airflow-up:
	docker compose -f orchestration/docker-compose.yaml up -d

airflow-down:
	docker compose -f orchestration/docker-compose.yaml down

# --- Demo (PBI 25b) -----------------------------------------------------------
# Static demo:  make demo
# Live demo:    make demo DEMO_MODE=1
#   Live mode brings up the Airflow stack and triggers refresh_15min_dag every
#   2 seconds (simulating 15 min at 450x speedup). Press Ctrl-C to stop.
# Teardown:     make demo-stop

COMPOSE_FILE = orchestration/docker-compose.yaml

demo:
ifdef DEMO_MODE
	@echo "[demo] DEMO_MODE=$(DEMO_MODE) -- live cadence demo"
	@test -f orchestration/.env || { echo "ERROR: orchestration/.env not found. Run: cp orchestration/.env.example orchestration/.env"; exit 1; }
	DEMO_MODE=$(DEMO_MODE) docker compose -f $(COMPOSE_FILE) up -d
	@echo "[demo] Waiting for Airflow scheduler..."
	@sleep 5
	@echo "[demo] Starting external trigger loop (Ctrl-C to stop)..."
	@trap 'echo " [demo] Stopped."; exit 0' INT; \
	while true; do \
		docker compose -f $(COMPOSE_FILE) exec -T airflow-scheduler airflow dags trigger refresh_15min_dag || true; \
		sleep 2; \
	done
else
	@echo "[demo] Static demo -- serving cached predictions"
	uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000
endif

demo-stop:
	docker compose -f $(COMPOSE_FILE) down

.PHONY: setup run mlflow test clean remove airflow-init airflow-up airflow-down demo demo-stop
