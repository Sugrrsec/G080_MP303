
import json
import logging
import sys
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

import cloudpickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "model.pkl"

model = None


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": "mp303-api",
            "event": record.getMessage()
        }

        if hasattr(record, "prediction"):
            log_data["prediction"] = record.prediction

        return json.dumps(log_data)


logger = logging.getLogger("mp303.api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

logger.propagate = False


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

    logger.info("model_loaded")

    yield

    model = None
    logger.info("model_unloaded")


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
    prediction = float(model.predict(input_data)[0])

    logger.info(
        "prediction_generated",
        extra={"prediction": prediction}
    )

    return {"prediction": prediction}
