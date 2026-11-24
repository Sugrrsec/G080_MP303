
from contextlib import asynccontextmanager
from pathlib import Path

import cloudpickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "model.pkl"

model = None


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_exp_yr: float = Field(le=50)
    rain_mm_week: float
    slope_deg: float
    temp_c: float = Field(ge=5)
    bush_age_yr: float
    hours_worked: float
    flush: str
    section: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    with MODEL_PATH.open("rb") as file:
        model = cloudpickle.load(file)

    yield

    model = None


app = FastAPI(
    title="MP303 Tea Leaf Plucking API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_data)[0]

    return {"prediction": float(prediction)}
