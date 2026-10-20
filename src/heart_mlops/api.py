"""FastAPI model service with health, prediction, metrics, and request logging."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from heart_mlops.data import FEATURES

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
logger = logging.getLogger("heart_mlops.api")
REQUESTS = Counter("heart_api_requests_total", "API requests", ["method", "path", "status"])
LATENCY = Histogram("heart_api_request_duration_seconds", "API request duration", ["path"])
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/heart_disease_model.joblib"))


class PatientFeatures(BaseModel):
    """Validated feature payload; categorical UCI fields use numeric category codes."""

    model_config = ConfigDict(extra="forbid")
    age: Annotated[float, Field(ge=0, le=120)]
    sex: Annotated[int, Field(ge=0, le=1)]
    cp: Annotated[int, Field(ge=1, le=4)]
    trestbps: Annotated[float, Field(gt=0, le=300)]
    chol: Annotated[float, Field(ge=0, le=1000)]
    fbs: Annotated[int, Field(ge=0, le=1)]
    restecg: Annotated[int, Field(ge=0, le=2)]
    thalach: Annotated[float, Field(gt=0, le=300)]
    exang: Annotated[int, Field(ge=0, le=1)]
    oldpeak: Annotated[float, Field(ge=0, le=20)]
    slope: Annotated[int, Field(ge=1, le=3)]
    ca: Annotated[int, Field(ge=0, le=3)]
    thal: Annotated[int, Field(ge=3, le=7)]


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None
    yield


app = FastAPI(title="Heart Risk Classifier", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe_request(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        path = request.url.path
        elapsed = time.perf_counter() - started
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            path,
            status,
            elapsed * 1000,
        )


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded; train or mount the model artifact.")
    return {"status": "ok", "model": "loaded"}


@app.post("/predict")
def predict(payload: PatientFeatures, request: Request) -> dict[str, object]:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded; train or mount the model artifact.")
    features = pd.DataFrame([payload.model_dump()], columns=FEATURES)
    prediction = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0][prediction])
    return {
        "prediction": prediction,
        "label": "disease_present" if prediction else "disease_absent",
        "confidence": round(confidence, 4),
        "disclaimer": "Educational demonstration only; not a medical diagnosis.",
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)