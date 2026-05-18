"""New York Taxi Demand Forecasting — application package.

Top-level subpackages:
    core:        Cross-cutting infrastructure (config, logging, paths).
    domain:      Pure types (Pydantic schemas, enums) shared across layers.
    pipeline:    Batch data pipeline (Bronze → Silver → Gold).
    training:    Model training service.
    inference:   Prediction service (loads model, scores features).
    api:         FastAPI HTTP layer (routers, dependencies).
    monitoring:  Model/data monitoring (Evidently reports, drift).
"""
