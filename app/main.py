import random
from datetime import datetime
from functools import wraps
from http import HTTPStatus

from fastapi import FastAPI, Request

from app import __version__
from app.schemas import AccelerometerData

app = FastAPI(
    title="Athology Backend and ML Web Services",
    description="Exposes several REST API endpoints for computation and prediction on accelerometer data.",
    version=__version__,
)


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
    jumping (`1`) or not (`0`) at each timestep. The predictions are available at
    `response["data"]["is_jumping"]`.
    """
    is_jumping = random.randint(0, 1)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"is_jumping": is_jumping},
    }
    return response
