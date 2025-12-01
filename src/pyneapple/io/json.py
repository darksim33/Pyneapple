import json
from pathlib import Path
from typing import Any

import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Path and numpy array objects."""

    def default(self, obj) -> dict[str, str] | dict[str, str] | Any:
        """Override default method to handle custom types.

        Args:
            obj: Object to encode

        Returns:
            Serializable representation of the object
        """
        if isinstance(obj, Path):
            return {"__type__": "Path", "value": str(obj)}
        elif isinstance(obj, np.ndarray):
            _obj = obj.squeeze()
            if _obj.ndim == 1:
                # Transform 1D array to list
                return _obj.tolist()
            else:
                return {
                    "__type__": "ndarray",
                    "value": obj.tolist(),
                    "dtype": str(obj.dtype),
                    "shape": obj.shape,
                }
        return super().default(obj)


def custom_json_decoder(_dict):
    """Custom JSON decoder that reconstructs Path and numpy array objects.

    Args:
        dct: Dictionary to decode

    Returns:
        Decoded object (Path, ndarray, or original dict)
    """
    if "__type__" in _dict:
        if _dict["__type__"] == "Path":
            return Path(_dict["value"])
        elif _dict["__type__"] == "ndarray":
            array = np.array(_dict["value"], dtype=_dict["dtype"])
            return array.reshape(_dict["shape"])
    return _dict


def load_from_json(file: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        file (Path): The path to the JSON file.

    Returns:
        dict[str, Any]: The dictionary loaded from the JSON file.
    """
    with open(file, "r") as f:
        data = json.load(f, object_hook=custom_json_decoder)
    return data


def save_to_json(data: dict[str, Any], file_path: Path) -> None:
    """Save a dictionary to a JSON file.

    Args:
        data (dict[str, Any]): The dictionary to save.
        file (Path): The path to the JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, cls=CustomJSONEncoder)
