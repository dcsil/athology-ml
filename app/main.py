from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import List

import numpy as np
import tensorflow as tf
import typer
from fastapi import FastAPI, Request
from ml.jump_detection.model import FeatureExtractor

from app import __version__
from app.schemas import AccelerometerData, Model

app = FastAPI(
    title="Athology Backend and ML Web Services",
    description="Exposes several REST API endpoints for computation and prediction on accelerometer data.",
    version=__version__,
)

try:
    import sentry_sdk
    from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

    sentry_sdk.init(dsn="https://0c859b2275af41cf9f37eef75d2319ff@o358880.ingest.sentry.io/5603816")
    app.add_middleware(SentryAsgiMiddleware)
except ImportError:
    typer.secho(
        "sentry-sdk is not installed. Not monitoring exceptions.", fg=typer.colors.yellow, bold=True
    ),


model = Model()


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


def _make_prediction(accelerometer_data: AccelerometerData) -> List[bool]:
    input_data = np.fromiter(
        accelerometer_data.x + accelerometer_data.y + accelerometer_data.z, dtype=np.int32
    ).reshape(1, -1, 3)
    is_jumping = (model.model.predict(input_data) >= 0.5).reshape(-1).tolist()
    return is_jumping


@app.on_event("startup")
def app_startup():
    model.model = tf.keras.models.load_model(
        "pretrained_models/jump_detection.h5", custom_objects={"FeatureExtractor": FeatureExtractor()}
    )


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request):
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/jump-detection", tags=["Jump Detection"])
@construct_response
def _jump_detection(request: Request, accelerometer_data: AccelerometerData):
    """Given one or more timesteps of accelerometer data, predicts whether the athelete is
    jumping (`True`) or not (`False`) at each timestep. The predictions are available at
    `response["data"]["is_jumping"]`.
    """
    is_jumping = _make_prediction(accelerometer_data)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"is_jumping": is_jumping},
    }
    return response
