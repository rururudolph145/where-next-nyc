# Custom Airflow image for the EXP experiment stack.
#
# EXP-02 is the first step that runs the real app/ code (Ingestion) inside a
# task, so the stock apache/airflow image is fattened with the ingest deps
# (pandas/numpy/pyarrow/pyyaml). Baking them into the image — rather than a
# _PIP_ADDITIONAL_REQUIREMENTS runtime install — keeps container restarts fast
# and the environment reproducible (no re-download on every `up`).
#
# app/ itself is NOT copied in; it is bind-mounted read-only at runtime (see
# docker-compose-LocalExecutor.yml) so host edits are picked up without a
# rebuild. Only the dependencies live in the image.
FROM apache/airflow:2.10.3-python3.11

# pip runs as the unprivileged airflow user (UID 50000); installing as root is
# unsupported by the base image and breaks the entrypoint.
COPY requirements-exp.txt /requirements-exp.txt
RUN pip install --no-cache-dir -r /requirements-exp.txt
