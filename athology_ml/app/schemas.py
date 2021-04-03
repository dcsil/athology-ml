from typing import List

from pydantic import BaseModel, root_validator


class AccelerometerData(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]

    @root_validator
    def check_equal_num_timesteps(cls, values):
        x, y, z = values.get("x"), values.get("y"), values.get("z")
        len_x, len_y, len_z = len(x), len(y), len(z)
        if not len(x) == len(y) == len(z):
            raise ValueError(
                (
                    "x, y, and z timesteps must be of equal length."
                    f" Got {len_x}, {len_y}, and {len_z} respectively."
                )
            )
        return values

    class Config:
        # This will be auto-populated as an example in the docs.
        schema_extra = {
            "example": {
                "x": [0.477, 0.498, 0.292, 0.237, 0.198, 0.198, 0.137, 0.122, 0.165, 0.276, 0.296],
                "y": [
                    -0.002,
                    -0.688,
                    -0.595,
                    -0.478,
                    -0.428,
                    -0.368,
                    -0.289,
                    -0.291,
                    -0.229,
                    -0.25,
                    -0.281,
                ],
                "z": [-0.371, 0.537, 0.738, 0.566, 0.528, 0.531, 0.496, 0.507, 0.542, 0.554, 0.468],
            }
        }


class AthleteName(BaseModel):
    first: str
    last: str

    class Config:
        schema_extra = {
            "example": {
                "first": "John",
                "last": "Smith",
            }
        }


class AthleteJump(BaseModel):
    load: float
    height: float


class AthleteSession(BaseModel):
    date: str
    timestamp: str
    jumps: List[AthleteJump]
    num_jumps: int
    max_load: float
    max_height: float

    class Config:
        schema_extra = {
            "example": {
                "date": "30/01/2021",
                "timestamp": "1612021861",
                "jumps": [{"load": 4.67, "height": 1.23}],
                "num_jumps": 1,
                "max_load": 4.67,
                "max_height": 1.23,
            }
        }


class AthleteData(BaseModel):
    _id: str
    name: AthleteName
    # For enterprise customers, this is the email that registered this athlete.
    # For individual customers, this represents a personal email.
    email: str
    sessions: List[AthleteSession]

    class Config:
        schema_extra = {
            "example": {
                "_id": "28187aa4-89c5-11eb-8dcd-0242ac130003",
                "email": "johndoe@example.com",
                "name": {
                    "first": "Laurie",
                    "last": "Smith",
                },
                "sessions": [
                    {
                        "date": "30/01/2021",
                        "timestamp": "1612021861",
                        "jumps": [{"load": 4.67, "height": 1.23}],
                        "num_jumps": 1,
                        "max_load": 4.67,
                        "max_height": 1.23,
                    }
                ],
            }
        }
