FROM apache/airflow:2.10.3-python3.11

# pip runs as the unprivileged airflow user (UID 50000); installing as root is
# unsupported by the base image and breaks the entrypoint.
COPY requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt

# libgomp1 is required by LightGBM at runtime (OpenMP threading backend).
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow
