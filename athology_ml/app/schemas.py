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
