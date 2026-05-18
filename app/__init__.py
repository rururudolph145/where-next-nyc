"""New York Taxi Demand Forecasting — application package.

Top-level subpackages:
    core:        Cross-cutting infrastructure (config, logging, paths).
    domain:      Pure types (Pydantic schemas, enums) shared across layers.
    ingestion:   Ingestion library code (raw → Bronze → Silver → Gold).
    training:    Model training service.
    inference:   Prediction service (loads model, scores features).
    api:         FastAPI HTTP layer (routers, dependencies).
    monitoring:  Model/data monitoring (Evidently reports, drift).

End-to-end orchestration lives in ``pipeline/`` at the repo root (numbered
runner scripts that compose these subpackages).
"""
