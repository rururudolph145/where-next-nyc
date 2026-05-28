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
