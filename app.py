from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()


class InputData(BaseModel):
    hour: int
    day_of_week: int
    month: int
    is_weekend: int
    PULocationID: int
    passenger_count: float
    trip_distance: float


model = joblib.load('models/model.pkl')


@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}


@app.post("/predict")
async def predict(input_data: InputData):
    df = pd.DataFrame([input_data.model_dump().values()],
                      columns=input_data.model_dump().keys())
    pred = model.predict(df)
    return {"predicted_trip_count": float(pred[0])}
