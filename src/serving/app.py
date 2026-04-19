"""FastAPI serving endpoint for NYC Taxi demand predictions."""

from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class InputData(BaseModel):
    """Request schema for the prediction endpoint."""

    hour: int
    day_of_week: int
    month: int
    is_weekend: int
    PULocationID: int
    passenger_count: float
    trip_distance: float


model: Any = joblib.load("models/model.pkl")


@app.get("/")
async def read_root() -> dict[str, object]:
    """Health check endpoint.

    Returns:
        Status and model version information.
    """
    return {"health_check": "OK", "model_version": 1}


@app.post("/predict")
async def predict(input_data: InputData) -> dict[str, float]:
    """Generate a demand prediction for the given input features.

    Args:
        input_data: Feature values for a single prediction request.

    Returns:
        Dictionary containing the predicted trip count.
    """
    df = pd.DataFrame(
        [input_data.model_dump().values()],
        columns=input_data.model_dump().keys(),
    )
    pred = model.predict(df)
    return {"predicted_trip_count": float(pred[0])}
