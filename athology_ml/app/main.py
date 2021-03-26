from datetime import datetime
from functools import wraps
from http import HTTPStatus

import numpy as np
import pymongo
import typer
from athology_ml import __version__
from athology_ml.app.schemas import AccelerometerData, AthleteSession, AthleteName
from athology_ml.app.util import load_jump_detection_model
from bson.objectid import ObjectId
from fastapi import Depends, FastAPI, Request
from tensorflow.keras import Model

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
        "sentry-sdk is not installed. Not monitoring exceptions.", fg=typer.colors.YELLOW, bold=True
    ),


client = pymongo.MongoClient(
    "mongodb+srv://admin:Dvr4smYFR0vYjCEd@jump-detection.mwnew.mongodb.net/jump-detection?retryWrites=true&w=majority"
)
db = client["development"]
col = db["athletes"]


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


@app.get("/get-all-athletes", tags=["Database"])
@construct_response
def _get_all_athletes(request: Request):
    """Returns the `_id`, `first_name` and `last_name` for all athletes."""
    results = []
    for result in col.find({}, {"_id": 1, "name": 1}):
        result["_id"] = str(result["_id"])
        results.append(result)

    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": results}
    return response


@app.get("/get-athlete-by-id", tags=["Database"])
@construct_response
def _get_athlete_by_id(request: Request, _id: str):
    """Return the data for an athelete with id `_id`."""
    result = col.find_one(
        {"_id": ObjectId(_id)},
    )
    result["_id"] = str(result["_id"])

    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": result}
    return response


@app.post("/create-new-athlete", tags=["Database"])
@construct_response
def _create_new_athlete(request: Request, name: AthleteName):
    """Creates a new athlete with `name` returns their `_id`."""
    print(name.dict())
    result = col.insert_one({"name": name.dict()})

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"_id": str(result.inserted_id)},
    }

    return response


@app.post("/add-athlete-session", tags=["Database"])
@construct_response
def _add_athlete_session(request: Request, _id: str, athlete_session: AthleteSession):
    """Adds or updates the session data for the athlete with id `_id` with `athlete_session`."""
    athlete_session = athlete_session.dict()

    query = {"_id": ObjectId(_id)}
    result = col.find_one(query)

    if "sessions" in result:
        sessions = result["sessions"] + [athlete_session]
    else:
        sessions = [athlete_session]

    result = col.update_one(query, {"$set": {"sessions": sessions}})

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"acknowledged": result.acknowledged},
    }
    return response


@app.post("/jump-detection", tags=["Prediction"])
@construct_response
def _jump_detection(
    request: Request,
    accelerometer_data: AccelerometerData,
    model: Model = Depends(load_jump_detection_model),
):
    """Given one or more timesteps of accelerometer data, predicts whether the athlete is
    jumping (`True`) or not (`False`) at each timestep. The predictions are available at
    `response["data"]["is_jumping"]`.
    """
    # The data must be converted from a list of floats to a numpy array of the correct shape.
    # np.fromiter should be slightly faster than np.asarray.
    input_data = np.fromiter(
        accelerometer_data.x + accelerometer_data.y + accelerometer_data.z, dtype=np.int32
    ).reshape(1, -1, 3)
    is_jumping = (model.predict(input_data) >= 0.5).reshape(-1).tolist()

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"is_jumping": is_jumping},
    }
    return response


# %%
