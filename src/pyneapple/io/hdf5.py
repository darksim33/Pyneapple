"""HDF5 I/O handling for data storage and retrieval.

This module provides functionality to save and load Python dictionaries to/from HDF5 files
with special encoding for numpy arrays, Path objects, and lists. It preserves data types
and structure through recursive encoding/decoding.

Features:
    - Recursive dictionary to HDF5 conversion
    - Gzip-compressed storage for numpy arrays
    - Type preservation for non-string dictionary keys (int, tuple)
    - Automatic string encoding/decoding (bytes to UTF-8)
    - Path object serialization
    - List vs array distinction

Special Encodings:
    - numpy.ndarray: Stored as compressed dataset with '__type__' attribute
    - pathlib.Path: Stored as string with '__type__' marker
    - list: Stored with '__type__' marker to distinguish from arrays
    - int/tuple keys: Type preserved via '__name_type__' attribute

Functions:
    save_to_hdf5: Save dictionary to HDF5 file
    load_from_hdf5: Load HDF5 file to dictionary
    dict_to_hdf5: Recursively write dictionary to HDF5 group
    hdf5_to_dict: Recursively read HDF5 group to dictionary
    save_params_to_hdf5: Save fitted parameter maps to HDF5
    save_result_to_hdf5: Save a complete FitResult (params + diagnostics + metadata) to HDF5

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

import ast
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from loguru import logger

from ..result import FitResult

# --- Export

_DEFAULT_GZIP_LEVEL: int = 4
"""Default gzip compression level used when storing numpy arrays."""


def _encode_key(key: str | int | tuple) -> tuple[str, str]:
    """Key might be ints or tuple. Encoding to preserve type."""
    if isinstance(key, str):
        return key, "str"
    elif isinstance(key, int):
        key_type = "int"
        return str(key), key_type
    elif isinstance(key, tuple):
        key_type = "tuple"
        return str(key), key_type
    else:
        return str(key), "str"


def _create_group(name: str | int | tuple, group: h5py.Group) -> h5py.Group:
    """Perform key encoding on group creation."""
    key, key_type = _encode_key(name)
    subgroup = group.create_group(name=key)
    if not isinstance(name, str):
        subgroup.attrs["__name_type__"] = key_type
    return subgroup


def _create_dataset(name: str | int | tuple, data, group: h5py.Group) -> h5py.Dataset:
    """Perform key encoding on dataset creation."""
    key, key_type = _encode_key(name)
    dataset = group.create_dataset(name=key, data=data)
    if not isinstance(name, str):
        dataset.attrs["__name_type__"] = key_type
    return dataset


def _encode_array(array: np.ndarray[Any, Any], group: h5py.Group, **kwargs) -> None:
    """Encode numpy arrays with compression."""
    compression: str = kwargs.get("compression", "gzip")
    compression_opts: int = kwargs.get(
        "compression_opts", _DEFAULT_GZIP_LEVEL if compression == "gzip" else None
    )
    group.attrs["__type__"] = "np.ndarray"
    group.create_dataset(
        "data",
        data=array,
        compression=compression,
        compression_opts=compression_opts,
    )


def _encode_path(path_obj: Path, group: h5py.Group):
    """Encode Path variables."""
    group.attrs["__type__"] = "Path"
    group.create_dataset("path", data=path_obj.__str__())


def _encode_list(_list: list, group: h5py.Group) -> None:
    """Encode lists to distinguish them from arrays."""
    group.attrs["__type__"] = "list"
    group.create_dataset("data", data=_list)


def dict_to_hdf5(_dict: dict[str, Any], h5: h5py.Group, **kwargs: Any):
    """Recursively write dict to an HDF5 group/file.

    Args:
        _dict: Dictionary to write.
        h5: HDF5 group to write to.
        **kwargs: Additional keyword arguments:
            compression (str): Compression filter for numpy arrays.
            compression_opts (int): Level of compression.
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
        return ast.literal_eval(key)
    return key


def _decode_array(group: h5py.Group) -> np.ndarray[Any, Any]:
    """Decode compressed array back to np.ndarray"""
    return group["data"][:]


def _decode_path(group: h5py.Group) -> Path:
    """Decode Path objects"""
    path_str: str = group["path"][()]
    if isinstance(path_str, bytes):
        path_str = path_str.decode("utf-8")
    return Path(path_str)


def _decode_list(group: h5py.Group) -> list:
    """Decode lists back to lists instead of arrays."""
    data = group["data"][:]
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data if isinstance(data, list) else list(data)


def hdf5_to_dict(group: h5py.Group) -> dict[Any, Any]:
    """Load data from hdf5 to dict.

    Args:
        group: Group or file to load from.

    Returns:
        dict: Decoded dictionary with original key types and value types restored.
    """
    _dict = {}
    for key in group.keys():
        value = group[key]
        decoded_key = (
            _decode_key(key, value)
            if isinstance(value, (h5py.Group, h5py.Dataset))
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
                            data = data.decode("utf-8")
                    else:
                        # For arrays with 1 or more dimensions
                        data = np.array(
                            [
                                (
                                    item.decode("utf-8")
                                    if isinstance(item, bytes)
                                    else item
                                )
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
        filepath: Path to hdf5 file to load.

    Returns:
        dict: Dictionary holding loaded decoded data.
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as file:
        return hdf5_to_dict(file)


def save_params_to_hdf5(
    fitted_params: dict[str, np.ndarray],
    pixel_indices: list[tuple[int, ...]],
    spatial_shape: tuple[int, ...],
    file_path: Path | str,
) -> None:
    """Save fitted parameter maps to an HDF5 file.

    Each parameter is stored as a compressed spatial array (3-D or 4-D).
    Pixel indices and spatial shape are stored as metadata.

    Args:
        fitted_params: Dictionary of parameter name → 1-D (or 2-D) array of
            shape ``(n_pixels,)`` or ``(n_pixels, n_extra)``.
        pixel_indices: Spatial index for each pixel.
        spatial_shape: Spatial shape of the output volume, e.g. ``(X, Y, Z)``.
        file_path: Output ``.h5`` path. Parent directories are created if needed.

    Examples:
        >>> save_params_to_hdf5(fitter.fitted_params_, fitter.pixel_indices,
        ...                     fitter.image_shape[:3], "results.h5")
    """
    from .nifti import reconstruct_maps

    if not fitted_params:
        raise ValueError("fitted_params is empty — nothing to export.")

    maps = reconstruct_maps(fitted_params, pixel_indices, spatial_shape)
    data: dict[str, Any] = {
        "params": {k: v for k, v in maps.items()},
        "pixel_indices": np.array(pixel_indices, dtype=np.int32),
        "spatial_shape": np.array(spatial_shape, dtype=np.int32),
    }

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_hdf5(data, file_path)
    logger.info(f"Saved parameter maps to: {file_path}")


def save_result_to_hdf5(
    result: FitResult,
    spatial_shape: tuple[int, ...],
    file_path: Path | str,
) -> None:
    """Save a complete :class:`~pyneapple.result.FitResult` to an HDF5 file.

    Writes three top-level groups:

    * ``params/`` — one spatially-reconstructed 3-D array per fitted parameter.
    * ``diagnostics/`` — per-pixel quality metrics reconstructed to spatial
      volumes where possible:

      - ``success`` — float32 convergence map (1 = converged, 0 = not).
      - ``r_squared`` — per-pixel R² map.
      - ``residuals`` — per-pixel residual-norm map.
      - ``n_iterations`` — per-pixel iteration-count map.
      - ``covariance`` — flat ``(n_pixels, n_params, n_params)`` tensor
        (not reconstructed spatially due to its 3-D nature per pixel).

    * ``metadata/`` — scalar provenance fields:

      - ``fit_time``, ``solver_name``, ``model_name``, ``n_pixels``,
        ``convergence_rate``, ``mean_r_squared``, ``spatial_shape``,
        ``pixel_indices``, ``image_shape`` (when available).

    Fields that are ``None`` in the result are silently omitted.

    Args:
        result: The :class:`~pyneapple.result.FitResult` returned by a fitter.
        spatial_shape: 3-D spatial shape ``(X, Y, Z)`` of the original image.
        file_path: Output ``.h5`` path.  Parent directories are created if
            needed.

    Raises:
        ValueError: If ``result.params`` is empty.

    Examples:
        >>> save_result_to_hdf5(fitter.results_, image_data.shape[:3],
        ...                     "results_diagnostics.h5")
    """
    from .nifti import reconstruct_maps

    if not result.params:
        raise ValueError("FitResult.params is empty — nothing to export.")

    pixel_indices = result.pixel_indices

    # ------------------------------------------------------------------
    # Helper: scatter a flat per-pixel array back to a spatial volume
    # ------------------------------------------------------------------
    def _to_spatial(arr: np.ndarray, key: str, dtype=np.float32) -> np.ndarray:
        arr = arr.astype(dtype)
        if pixel_indices is not None:
            return reconstruct_maps({key: arr}, pixel_indices, spatial_shape)[key]
        return arr

    # ------------------------------------------------------------------
    # params/ — reconstructed spatial maps
    # ------------------------------------------------------------------
    if pixel_indices is not None:
        param_maps = reconstruct_maps(result.params, pixel_indices, spatial_shape)
    else:
        param_maps = dict(result.params)

    data: dict[str, Any] = {"params": param_maps}

    # ------------------------------------------------------------------
    # diagnostics/ — per-pixel quality metrics
    # ------------------------------------------------------------------
    diag: dict[str, Any] = {}

    if result.success is not None:
        diag["success"] = _to_spatial(result.success, "success", np.float32)
    if result.r_squared is not None:
        diag["r_squared"] = _to_spatial(result.r_squared, "r_squared", np.float32)
    if result.residuals is not None:
        diag["residuals"] = _to_spatial(result.residuals, "residuals", np.float32)
    if result.n_iterations is not None:
        diag["n_iterations"] = _to_spatial(
            result.n_iterations, "n_iterations", np.float32
        )
    if result.covariance is not None:
        # Stored flat: (n_pixels, n_params, n_params) — spatial reconstruction
        # is non-trivial for a 3-D per-pixel tensor.
        diag["covariance"] = result.covariance.astype(np.float32)

    if diag:
        data["diagnostics"] = diag

    # ------------------------------------------------------------------
    # metadata/ — scalars and provenance
    # ------------------------------------------------------------------
    meta: dict[str, Any] = {
        "fit_time": result.fit_time,
        "solver_name": result.solver_name,
        "model_name": result.model_name,
        "n_pixels": result.n_pixels,
        "convergence_rate": result.convergence_rate,
        "spatial_shape": np.array(spatial_shape, dtype=np.int32),
    }
    if result.mean_r_squared is not None:
        meta["mean_r_squared"] = result.mean_r_squared
    if pixel_indices is not None:
        meta["pixel_indices"] = np.array(pixel_indices, dtype=np.int32)
    if result.image_shape is not None:
        meta["image_shape"] = np.array(result.image_shape, dtype=np.int32)

    data["metadata"] = meta

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_hdf5(data, file_path)
    logger.info(f"Saved FitResult diagnostics to: {file_path}")
