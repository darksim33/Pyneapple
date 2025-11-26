"""HDF5 I/O handling for data storage and retrieval.

This module provides functionality to save and load Python dictionaries to/from HDF5 files
with special encoding for numpy arrays, Path objects, and lists. It preserves data types
and structure through recursive encoding/decoding.

Features:
    - Recursive dictionary to HDF5 conversion
    - Sparse array storage with compression for numpy arrays
    - Type preservation for non-string dictionary keys (int, tuple)
    - Automatic string encoding/decoding (bytes to UTF-8)
    - Path object serialization
    - List vs array distinction

Special Encodings:
    - numpy.ndarray: Stored as compressed sparse COO matrices with '__type__'
    - pathlib.Path: Stored as string with '__type__' marker
    - list: Stored with '__type__' marker to distinguish from arrays
    - int/tuple keys: Type preserved via '__name_type__' attribute

Functions:
    save_to_hdf5: Save dictionary to HDF5 file
    load_from_hdf5: Load HDF5 file to dictionary
    dict_to_hdf5: Recursively write dictionary to HDF5 group
    hdf5_to_dict: Recursively read HDF5 group to dictionary

Example:
    >>> from pathlib import Path
    >>> import numpy as np
    >>>
    >>> data = {
    ...     'array': np.array([[1, 2], [3, 4]]),
    ...     'path': Path('/some/path'),
    ...     'nested': {
    ...         'values': [1, 2, 3],
    ...         42: 'integer key'
    ...     }
    ... }
    >>>
    >>> # Save to HDF5
    >>> save_to_hdf5(data, 'output.h5', compression='gzip', compression_opts=9)
    >>>
    >>> # Load from HDF5
    >>> loaded = load_from_hdf5('output.h5')
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import sparse

from ..utils.logger import logger

# --- Export


def _encode_key(key: str | int | tuple) -> str:
    """Key might be ints or tuple. Encoding to preserve type."""
    if isinstance(key, str):
        return key, None
    elif isinstance(key, int):
        key_type = "int"
        return str(key), key_type
    elif isinstance(key, tuple):
        key_type = "tuple"
        return str(key), key_type


def _create_group(name: str | int | tuple, group: h5py.Group) -> h5py.Group:
    """Perform key encoding on group creation."""
    key, key_type = _encode_key(name)
    subgroup = group.create_group(name=key)
    if not isinstance(name, str):
        subgroup.attrs["__name_type__"] = key_type
    return subgroup


def _create_dataset(
    name: str | int | tuple, data, group: h5py.Group | h5py.Dataset
) -> h5py.Dataset:
    """Perform key encoding on dataset creation."""
    key, key_type = _encode_key(name)
    dataset = group.create_dataset(name=key, data=data)
    if not isinstance(name, str):
        dataset.attrs["__name_type__"] = key_type
    return dataset


def _encode_array(array: np.ndarray[Any, Any], group: h5py.Group, **kwargs) -> None:
    """Encode numpy arrays for sparsing and compression."""
    compression: str = kwargs.get("compression", "gzip")
    compression_opts: int = kwargs.get(
        "compression_opts", 4 if compression == "gzip" else None
    )
    # Convert to sparse
    sparse_array = sparse.COO.from_numpy(array)

    group.attrs["__type__"] = "np.ndarray"
    group.create_dataset(
        "data",
        data=sparse_array.data,
        compression=compression,
        compression_opts=compression_opts,
    )
    group.create_dataset(
        "coords",
        data=sparse_array.coords,
        compression=compression,
        compression_opts=compression_opts,
    )
    group.attrs["shape"] = sparse_array.shape


def _encode_path(path_obj: Path, group: h5py.Group):
    """Encode Path variables."""
    group.attrs["__type__"] = "Path"
    group.create_dataset("path", data=path_obj.__str__())


def _encode_list(_list: list, group: h5py.Group) -> None:
    "Encode lists to distinguish them from arrays"
    group.attrs["__type__"] = "list"
    group.create_dataset("data", data=_list)


def dict_to_hdf5(_dict: dict[str, Any], h5: h5py.Group, **kwargs: Any):
    """
    Recursivley write dict to an HDF5 group/fíle.

    Args:
        _dict (dict): Dictionary to write.
        h5 (h5py.Group): H5 Group to write to.
        **kwargs:
            compression (str): np.array compression
            compression_lvl (int): level of compression
    """
    for key, value in _dict.items():
        if isinstance(value, dict):
            subgroup = _create_group(key, h5)
            dict_to_hdf5(value, subgroup, **kwargs)
        elif isinstance(value, np.ndarray):
            subgroup = _create_group(key, group=h5)
            _encode_array(value, subgroup, **kwargs)
        elif isinstance(value, Path):
            subgroup = _create_group(name=key, group=h5)
            _encode_path(value, subgroup)
        elif isinstance(value, list):
            subgroup = _create_group(name=key, group=h5)
            _encode_list(value, subgroup)
        else:
            _create_dataset(key, value, h5)


def save_to_hdf5(data: dict[str, Any], filepath: Path | str, **kwargs) -> None:
    """Save dict to hdf5 file.

    Args:
        data (dict): Dictionary containing data to be saved.
        filepath (Path | str): Path to file to save to.
        **kwargs: Additional keyword arguments to pass to _encode_array.
            compression (str | None): Compression filter for arrays (default: gzip).
            compression_opts (int | tuple | None): Compression level for arrays.
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "w") as file:
        dict_to_hdf5(data, file, **kwargs)


# --- Import


def _decode_key(key: str, group: h5py.Group | h5py.Dataset) -> str | int | tuple:
    """Decode the key back to the original type using the '__name_type__' attribute"""
    if "__name_type__" not in group.attrs:
        return key
    name_type = group.attrs["__name_type__"]
    if name_type == "int":
        return int(key)
    elif name_type == "tuple":
        return eval(key)
    return key


def _decode_array(group: h5py.Group) -> np.ndarray[Any, Any]:
    """Decode sparse matrix back to np.ndarray"""
    data = group["data"][:]
    coords = group["coords"][:]
    shape = tuple(group.attrs["shape"])
    sparse_array = sparse.COO(coords=coords, data=data, shape=shape)
    return sparse_array.todense()


def _decode_path(group: h5py.Group) -> Path:
    """Decode Path objects"""
    path_str: str = group["path"][()]
    if isinstance(path_str, bytes):
        path_str = path_str.decode("utf-8")
    return Path(path_str)


def _decode_list(group: h5py.Group) -> list:
    "Decode lists back to lists instead of arrays"
    data = group["data"][:]
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data if isinstance(data, list) else list(data)


def hdf5_to_dict(group: h5py.Group) -> dict[Any, Any]:
    """Load data from hdf5 to dict.

    Args:
        group (h5py.Group | h5py.File): Group or File to load from.
    """
    _dict = {}
    for key in group.keys():
        value = group[key]
        decoded_key = (
            _decode_key(key, value)
            if isinstance(value, h5py.Group | h5py.Dataset)
            else key
        )
        if isinstance(value, h5py.Group):
            # Catch special encoded groups
            if "__type__" in value.attrs:
                _type = value.attrs["__type__"]
                if _type == "np.ndarray":
                    _dict[decoded_key] = _decode_array(value)
                elif _type == "Path":
                    _dict[decoded_key] = _decode_path(value)
                elif _type == "list":
                    _dict[decoded_key] = _decode_list(value)
                else:
                    _dict[decoded_key] = hdf5_to_dict(value)
            else:
                # continue with regular groups
                _dict[decoded_key] = hdf5_to_dict(value)

        elif isinstance(value, h5py.Dataset):
            # regular datasets
            data = value[()]
            # string decoding if needed
            if isinstance(data, bytes):
                try:
                    data = data.decode("utf-8")
                except Exception as e:
                    warn_msg = f"Unexpected error decoding array '{decoded_key}': {e}"
                    logger.warning(warn_msg)
            elif isinstance(data, np.ndarray) and data.dtype.kind in ["S", "O"]:
                # data.dtype.kind in ['S', 'O'] - Is it a string type?
                #   'S' = byte string (fixed-length, like b'hello')
                #   'O' = object dtype (can contain variable-length strings)
                try:
                    if data.ndim == 0:
                        # ndim == 0 means it's a scalar wrapped in an array
                        data = data.item()
                        if isinstance(data, bytes):
                            data.decode("utf-8")
                    else:
                        # For arrays with 1 or more dimensions
                        data = np.array(
                            [
                                item.decode("utf-8")
                                if isinstance(item, bytes)
                                else item
                                for item in data.flat
                            ]
                        ).reshape(data.shape)
                except Exception as e:
                    warn_msg = f"Unexpected error decoding array '{decoded_key}': {e}"
                    logger.warning(warn_msg)
            _dict[decoded_key] = data

    return _dict


def load_from_hdf5(filepath: Path | str) -> dict[str, Any]:
    """Load hdf5 file to dictionary.

    Args:
        filepath (Path | str): Path to hdf5 file to load
    Returns:
        dict (dict): Dictionary holding loaded decoded data.
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as file:
        return hdf5_to_dict(file)
