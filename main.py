"""Backwards-compatibility shim — delegates to ``pipeline.run_all``.

The canonical DAG runner lives in :mod:`pipeline.run_all`. This file exists
only so legacy invocations (``python main.py``, ``make run``) keep working
during the ``src/`` → ``app/`` migration. New code and docs should reference
``pipeline.run_all`` directly:

    uv run python -m pipeline.run_all
"""

from pipeline.run_all import run_all

if __name__ == "__main__":
    run_all()
