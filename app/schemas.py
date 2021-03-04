from typing import List

import tensorflow as tf
from pydantic import BaseModel, root_validator


class Model(BaseModel):
    model: tf.keras.Model = None

    class Config:
        arbitrary_types_allowed = True


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
