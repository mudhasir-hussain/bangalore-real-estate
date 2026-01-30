import os
import json
import pickle
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression


__data_columns: Optional[List[str]] = None
__locations: Optional[List[str]] = None
__model: Optional[LinearRegression] = None


def load_saved_artifacts() -> None:
    global __data_columns, __locations, __model

    print("loading saved artifacts...start")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    columns_path = os.path.join(base_dir, "artifacts", "columns.json")
    model_path = os.path.join(base_dir, "artifacts", "banglore_home_prices_model.pickle")

    with open(columns_path, "r") as f:
        data = json.load(f)["data_columns"]
        __data_columns = data
        __locations = data[3:]  

    if __model is None:
        with open(model_path, "rb") as f:
            __model = pickle.load(f)

    print("loading saved artifacts...done")


def get_estimated_price(
    location: str,
    sqft: float,
    bhk: int,
    bath: int
) -> float:

    if __data_columns is None or __model is None:
        raise RuntimeError("Artifacts not loaded. Call load_saved_artifacts() first.")

    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns), dtype=np.float64)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    x = x.reshape(1, -1)  

    return round(float(__model.predict(x)[0]), 2)


def get_location_names() -> List[str]:
    if __locations is None:
        raise RuntimeError("Artifacts not loaded.")
    return __locations


def get_data_columns() -> List[str]:
    if __data_columns is None:
        raise RuntimeError("Artifacts not loaded.")
    return __data_columns


if __name__ == "__main__":
    load_saved_artifacts()

    print(get_location_names())
    print(get_estimated_price("1st Phase JP Nagar", 1000, 3, 3))
    print(get_estimated_price("1st Phase JP Nagar", 1000, 2, 2))
    print(get_estimated_price("Kalhalli", 1000, 2, 2))
    print(get_estimated_price("Ejipura", 1000, 2, 2))
