import os
import pickle
from typing import TypeVar


def save_pickle(state: object, name: str) -> None:
    pickle.dump(state, open(name, "wb"))


T = TypeVar("T")


def load_pickle(name: str, obj_type: type[T]) -> T | None:
    if not os.path.exists(name):
        raise FileNotFoundError(f"Pickle file {name} does not exist.")
    ret = pickle.load(open(name, "rb"))
    assert isinstance(ret, obj_type), f"Expected type {obj_type}, got {type(ret)}"
    return ret
