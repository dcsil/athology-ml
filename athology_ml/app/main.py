import os
from datetime import datetime
from functools import wraps
from http import HTTPStatus

import numpy as np
import pymongo
from athology_ml import __version__, msg
from athology_ml.app import util
from athology_ml.app.schemas import AccelerometerData, AthleteName, AthleteSession, User
from bson.objectid import ObjectId
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_login import LoginManager
from fastapi_login.exceptions import InvalidCredentialsException
from tensorflow.keras import Model

DB_USER = os.environ.get("DB_USER", "development")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "development")
SENTRY_SDK_DSN = os.environ.get("SENTRY_SDK_DSN")
LOGIN_MANAGER_SECRET = os.environ.get("LOGIN_MANAGER_SECRET")

app = FastAPI(
    title="Athology Backend and ML Web Services",
    description="Exposes several REST API endpoints for computation and prediction on accelerometer data.",
    version=__version__,
)

try:
    import sentry_sdk
    from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

    if SENTRY_SDK_DSN:
        sentry_sdk.init(dsn=SENTRY_SDK_DSN)
        app.add_middleware(SentryAsgiMiddleware)
    else:
        msg.warn("sentry-sdk DSN not provided. is not installed. Not monitoring exceptions.")
except ImportError:
    msg.warn("sentry-sdk is not installed. Not monitoring exceptions.")


origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = pymongo.MongoClient(
    f"mongodb+srv://{DB_USER}:{DB_PASSWORD}@jump-detection.mwnew.mongodb.net/jump-detection?retryWrites=true&w=majority"
)

database = "development" if DB_USER == "development" else "production"
db = client[database]
users_col = db["users"]
athletes_col = db["athletes"]

if not LOGIN_MANAGER_SECRET:
    LOGIN_MANAGER_SECRET = os.urandom(24).hex()
    msg.warn(
        f"LOGIN_MANAGER_SECRET enviornment variable not found. Defaulting to: {LOGIN_MANAGER_SECRET}"
    )
manager = LoginManager(LOGIN_MANAGER_SECRET, token_url="/auth/login")


@manager.user_loader
def _load_user(email: str) -> User:
    user = users_col.find_one(
        {"email": email},
    )
    return user


@app.post("/auth/signup", tags=["Authentication"])
def _signup(data: OAuth2PasswordRequestForm = Depends()):
    """Sign up with the provided user and password, if they don't already exist."""
    email = data.username
    password = data.password
    user = _load_user(email)

    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": {}}

    if user:
        response["message"] = "An account for that email address already exists"
        response["status-code"] = HTTPStatus.BAD_REQUEST
    else:
        salt, key = util.salt_password(password)
        result = users_col.insert_one({"email": email, "salt": salt, "key": key})
        response["data"] = ({"acknowledged": result.acknowledged},)

    return response


@app.post("/auth/login", tags=["Authentication"])
def _login(data: OAuth2PasswordRequestForm = Depends()):
    """Login with the provided user and password, if they exist."""
    email = data.username
    password = data.password

    user = _load_user(email)
    if not user:
        raise InvalidCredentialsException

    _, key = util.salt_password(password, user["salt"])
    if key != user["key"]:
        raise InvalidCredentialsException

    access_token = manager.create_access_token(data=dict(sub=email))
    return {"access_token": access_token, "token_type": "bearer"}


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
def _get_all_athletes(request: Request, user=Depends(manager)):
    """Returns the `_id`, `first_name` and `last_name` for all athletes."""
    results = []
    for result in athletes_col.find({"email": user["email"]}, {"_id": 1, "name": 1}):
        result["_id"] = str(result["_id"])
        results.append(result)

    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": results}
    return response


@app.get("/get-athlete-by-id", tags=["Database"])
@construct_response
def _get_athlete_by_id(request: Request, _id: str, user=Depends(manager)):
    """Return the data for an athelete with id `_id`."""
    result = athletes_col.find_one(
        {"_id": ObjectId(_id), "email": user["email"]},
    )
    result["_id"] = str(result["_id"])

    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": result}
    return response


@app.post("/create-new-athlete", tags=["Database"])
@construct_response
def _create_new_athlete(request: Request, name: AthleteName, user=Depends(manager)):
    """Creates a new athlete with `name` returns their `_id`."""
    result = athletes_col.insert_one({"name": name.dict(), "email": user["email"]})

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"_id": str(result.inserted_id)},
    }

    return response


@app.post("/add-athlete-session", tags=["Database"])
@construct_response
def _add_athlete_session(
    request: Request, _id: str, athlete_session: AthleteSession, user=Depends(manager)
):
    """Adds or updates the session data for the athlete with id `_id` with `athlete_session`."""
    athlete_session = athlete_session.dict()

    query = {"_id": ObjectId(_id), "email": user["email"]}
    result = athletes_col.find_one(query)

    if "sessions" in result:
        sessions = result["sessions"] + [athlete_session]
    else:
        sessions = [athlete_session]

    result = athletes_col.update_one(query, {"$set": {"sessions": sessions}})

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
    model: Model = Depends(util.load_jump_detection_model),
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
    is_jumping = util.filter_predictions(is_jumping)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"is_jumping": is_jumping},
    }
    return response
