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

.PHONY: setup run mlflow test clean remove airflow-init airflow-up airflow-down
