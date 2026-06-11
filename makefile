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
# COMPOSE_FILES bundles the upstream-verbatim base file with the local
# docker-compose.override.yaml (mounts app/ and orchestration/ into the
# containers, sets PYTHONPATH). Both must be passed explicitly because
# Compose only auto-discovers the override file in the cwd, not next to
# the file given via -f.
COMPOSE_FILES = -f orchestration/docker-compose.yaml -f orchestration/docker-compose.override.yaml

airflow-init:
	docker compose $(COMPOSE_FILES) up airflow-init

airflow-up:
	docker compose $(COMPOSE_FILES) up -d

airflow-down:
	docker compose $(COMPOSE_FILES) down

# --- Demo (PBI 25b) -----------------------------------------------------------
# Static demo:  make demo
# Live demo:    make demo DEMO_MODE=1
#   Live mode brings up the Airflow stack and triggers refresh_15min_dag every
#   2 seconds (simulating 15 min at 450x speedup). Press Ctrl-C to stop.
# Teardown:     make demo-stop

demo:
ifdef DEMO_MODE
	@echo "[demo] DEMO_MODE=$(DEMO_MODE) -- live cadence demo"
	@test -f orchestration/.env || { echo "ERROR: orchestration/.env not found. Run: cp orchestration/.env.example orchestration/.env"; exit 1; }
	DEMO_MODE=$(DEMO_MODE) docker compose $(COMPOSE_FILES) up -d
	@echo "[demo] Waiting for Airflow scheduler..."
	@sleep 5
	@echo "[demo] Starting FastAPI server in background..."
	@DEMO_MODE=$(DEMO_MODE) uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 > /tmp/refresh_15min_demo_uvicorn.log 2>&1 & \
	UVICORN_PID=$$!; \
	echo "[demo] uvicorn PID=$$UVICORN_PID (log: /tmp/refresh_15min_demo_uvicorn.log)"; \
	echo "[demo] Starting external trigger loop (Ctrl-C to stop)..."; \
	trap "echo ' [demo] Stopped.'; kill $$UVICORN_PID 2>/dev/null; exit 0" INT; \
	while true; do \
		docker compose $(COMPOSE_FILES) exec -T airflow-scheduler airflow dags trigger refresh_15min_dag || true; \
		sleep 2; \
	done
else
	@echo "[demo] Static demo -- serving cached predictions"
	uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000
endif

demo-stop:
	docker compose $(COMPOSE_FILES) down
	@pkill -f "uvicorn app.api.main:app" || true

.PHONY: setup run mlflow test clean remove airflow-init airflow-up airflow-down demo demo-stop
