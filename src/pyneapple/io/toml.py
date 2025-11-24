import sys
from pathlib import Path
from typing import Any

import numpy as np

# Import appropriate TOML library based on Python version
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from ..utils.logger import logger


def prepare_for_toml(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively prepare data for TOML serialization by converting custom types.

    Args:
        data: Dictionary to prepare

    Returns:
        Dictionary with custom types converted to TOML-compatible types
    """
    if isinstance(data, dict):
        return {key: prepare_for_toml(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [prepare_for_toml(item) for item in data]
    elif isinstance(data, Path):
        return {"__type__": "Path", "value": str(data)}
    elif isinstance(data, np.ndarray):
        return {
            "__type__": "ndarray",
            "value": data.tolist(),
            "dtype": str(data.dtype),
            "shape": list(data.shape),
        }
    else:
        return data


def reconstruct_from_toml(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively reconstruct custom types from TOML data.

    Args:
        data: Dictionary loaded from TOML

    Returns:
        Dictionary with custom types reconstructed
    """
    if isinstance(data, dict):
        # Check if this is a custom type marker
        if (
            "__type__" in data and len(data) <= 4
        ):  # Custom type dicts have specific keys
            if data["__type__"] == "Path":
                return Path(data["value"])
            elif data["__type__"] == "ndarray":
                array = np.array(data["value"], dtype=data["dtype"])
                return array.reshape(data["shape"])
        # Otherwise, recursively process all items
        return {key: reconstruct_from_toml(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [reconstruct_from_toml(item) for item in data]
    else:
        return data


def toml_dump(data, file_obj):
    """
    Dump data as TOML to a file object, using the appropriate library based on Python version.

    Args:
        data (dict): The data to serialize as TOML
        file_obj: A file-like object opened in the appropriate mode

    Raises:
        ImportError: If the required TOML writing library is not available
    """
    # Prepare data by converting custom types
    prepared_data = prepare_for_toml(data)

    if sys.version_info >= (3, 11):
        try:
            import tomlkit

            toml_string = tomlkit.dumps(prepared_data)
            file_obj.write(toml_string.encode("utf-8"))
        except ImportError:
            raise ImportError(
                "tomlkit library is required for writing TOML files in Python 3.11+. "
                "Please install it with 'pip install tomlkit'"
            )
    else:
        try:
            import tomli_w

            tomli_w.dump(prepared_data, file_obj)
        except ImportError:
            raise ImportError(
                "tomli-w library is required for writing TOML files in Python < 3.11. "
                + "Please install it with 'pip install tomli-w'"
            )


def load_from_toml(file_path: Path) -> dict[str, Any]:
    """
    Load data from a TOML file, using the appropriate library based on Python version.

    Args:
        file_path (Path): The path to the TOML file

    Returns:
        dict: The data loaded from the TOML file

    Raises:
        ImportError: If the required TOML reading library is not available
    """
    try:
        with file_path.open("rb") as file:
            data = tomllib.load(file)
        return reconstruct_from_toml(data)
    except tomllib.TOMLDecodeError as e:
        error_msg = f"Failed to parse TOML file {file_path}: {e}"
        logger.error(error_msg)
        raise tomllib.TOMLDecodeError(error_msg)


def save_to_toml(data: dict[str, Any], file_path: Path):
    """
    Save data to a TOML file, using the appropriate library based on Python version.

    Args:
        data (dict): The data to serialize as TOML
        file_path (Path): The path to the TOML file

    Raises:
        ImportError: If the required TOML writing library is not available
    """
    with file_path.open("wb") as file:
        toml_dump(data, file)
