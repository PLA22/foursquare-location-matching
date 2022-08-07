import json

import pandas as pd


def load_config(path):
    with open(path) as f:
        config = json.load(f)
    return config


def load_locations(path, **kwargs):
    dtype = {
        "id": "str",
        "name": "str",
        "latitude": "float64",
        "longitude": "float64",
        "address": "str",
        "city": "str",
        "state": "str",
        "zip": "str",
        "country": "str",
        "url": "str",
        "phone": "str",
        "categories": "str",
        "point_of_interest": "str"
    }
    return pd.read_csv(path, dtype=dtype, **kwargs)
