import numpy as np
from typing import Any

from loguru import logger

# CurveFit specific validation and transformation utilities


def validate_fixed_params(
    fixed_params: dict[str, Any],
    all_param_names: list[str],
) -> None:
    """Validate that fixed parameter names are valid model parameters.

    Args:
        fixed_params: Dictionary ``{name: value}`` of parameters to hold
            constant during fitting.
        all_param_names: Full ordered list of the model's parameter names
            (including those that may be fixed).

    Raises:
        ValueError: If any key in *fixed_params* is not in *all_param_names*.
        ValueError: If fixing all parameters would leave nothing to fit.
    """
    if not fixed_params:
        return

    unknown = set(fixed_params.keys()) - set(all_param_names)
    if unknown:
        raise ValueError(
            f"Unknown fixed parameter(s): {sorted(unknown)}. "
            f"Valid names: {all_param_names}"
        )

    if len(fixed_params) >= len(all_param_names):
        raise ValueError(
            "Cannot fix all parameters — at least one must remain free. "
            f"Fixed: {sorted(fixed_params.keys())}, all: {all_param_names}"
        )


def validate_fixed_param_maps(
    fixed_param_maps: dict[str, np.ndarray],
    spatial_shape: tuple[int, ...],
    all_param_names: list[str],
) -> None:
    """Validate per-pixel fixed parameter maps.

    Args:
        fixed_param_maps: ``{name: array}`` where each array must match
            *spatial_shape*.
        spatial_shape: Expected spatial dimensions ``(X, Y, Z)``.
        all_param_names: Full ordered list of the model's parameter names.

    Raises:
        ValueError: If a name is not a valid parameter, or an array's shape
            does not match *spatial_shape*.
    """
    validate_fixed_params(fixed_param_maps, all_param_names)

    for name, arr in fixed_param_maps.items():
        if arr.shape != spatial_shape:
            raise ValueError(
                f"Fixed param map '{name}' has shape {arr.shape}, "
                f"expected spatial shape {spatial_shape}."
            )


def validate_xdata(xdata: np.ndarray):
    """Validate xdata for curve fitting.

    Args:
        xdata: 1D array of independent variable (e.g., b-values).
    """
    if xdata.ndim != 1:
        raise ValueError(f"xdata must be a 1D array, but got shape {xdata.shape}.")


# Validate shapes of xdata and ydata for curve fitting


def validate_data_shapes(xdata: np.ndarray, ydata: np.ndarray):
    """Validate shapes of xdata and ydata for curve fitting.

    Args:
        xdata: 1D array of independent variable (e.g., b-values).
        ydata: 1D or 2D array of observed signals (shape: [n_voxels, n_xdata] or [img_shape, n_xdata]).

    Raises:
        ValueError: If shapes are inconsistent or invalid for curve fitting.

    Example:
        >>> xdata = np.array([0.1, 0.2, 0.3])
        >>> ydata = np.array([[1.0, 1.1, 1.2], [0.9, 1.0, 1.1]])
        >>> validate_data_shapes(xdata, ydata)
    """
    if xdata.ndim != 1:
        raise ValueError(f"xdata must be a 1D array, but got shape {xdata.shape}.")
    if ydata.ndim == 1:
        if ydata.shape[0] != xdata.shape[0]:
            raise ValueError(
                f"ydata length {ydata.shape[0]} does not match xdata length {xdata.shape[0]}."
            )
    elif ydata.ndim >= 2:
        if ydata.shape[-1] != xdata.shape[0]:
            raise ValueError(
                f"ydata second dimension {ydata.shape[-1]} does not match xdata length {xdata.shape[0]}."
            )
    else:
        raise ValueError(f"ydata must be 1D or 2D array, but got shape {ydata.shape}.")


# Validate segmentation matches image shape


def validate_segmentation(segmentation: np.ndarray, image_shape: tuple[int, int]):
    """Validate that segmentation matches image shape.

    Args:
        segmentation: 2D array of segmentation labels.
        image_shape: Expected shape of the image (height, width).

    Raises:
        ValueError: If segmentation shape does not match image shape.

    Example:
        >>> segmentation = np.array([[1, 1, 0], [0, 1, 1]])
        >>> image_shape = (2, 3)
        >>> validate_segmentation(segmentation, image_shape)
    """
    if segmentation.ndim != len(image_shape) - 1:
        # segmetation should have one less dimension than image shape (no channel dimension)
        if segmentation.shape[-1] == 1:
            # if segmentation has a singleton channel dimension
            logger.warning(
                f"Segmentation has an unexpected channel dimension with shape {segmentation.shape}. Squeezing to remove singleton dimension."
            )
            segmentation = np.squeeze(segmentation, axis=-1)
        else:
            raise ValueError(
                f"Segmentation must have one less dimension than image shape {image_shape}, but got shape {segmentation.shape}."
            )
    if segmentation.shape != image_shape[:-1]:
        raise ValueError(
            f"Segmentation shape {segmentation.shape} does not match expected image shape {image_shape[:-1]}."
        )
    return segmentation


# Validate parameter names and bounds for curve_fit
def validate_parameter_names(parameters: dict[str, Any], param_names: list[str]):
    """Validate parameter names against the model's expected parameters.

    Checks that all required parameter names are present in the parameters dict.

    Args:
        parameters: Dictionary of parameter values keyed by name.
        param_names: List of required parameter names.
    Raises:
        ValueError: If parameter names are missing or extra.
    """
    missing = set(param_names) - set(parameters.keys())
    if missing:
        raise ValueError(
            f"Missing bounds for required parameters: {missing}. "
            f"Required: {param_names}"
        )

    extra = set(parameters.keys()) - set(param_names)
    if extra:
        logger.warning(f"Extra bounds will be ignored: {extra}")


# transform p0 dictionary to numpy array in the correct order
def transform_p0(
    p0: dict[str, float], param_names: list[str], n_pixels: int | None = None
) -> np.ndarray:
    """Transform p0 dictionary to numpy array in the correct order.

    Args:
        p0: Dictionary of initial parameter guesses as {param: value}.
        param_names: List of required parameter names to ensure correct ordering.
        n_pixels: Number of pixels for which to transform p0.
    Returns:
        np.ndarray: Initial parameter guesses as a numpy array in the correct order.
    """
    validate_parameter_names(p0, param_names)
    if isinstance(p0[param_names[0]], (int, float)):
        p0_array = np.array(
            [p0[name] for name in param_names], dtype=float
        )  # Shape: (n_param,)
        if n_pixels is None:
            return p0_array  # Shape: (n_param,)
        else:
            return np.tile(
                p0_array[:, np.newaxis], (1, n_pixels)
            )  # Shape: (n_param, n_pixel)
    else:
        raise ValueError(
            "p0 must be provided as a dictionary with parameter names as keys and float values."
        )


# transform p0 dictionary to numpy array in the correct order for spatial fitting
def transform_p0_spatial(
    p0: dict[str, np.ndarray],
    param_names: list[str],
    image_shape: tuple[int, int],
    pixel_indices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Transform spatial p0 dictionary to numpy array in the correct order.

    Args:
        p0: Dictionary of initial parameter guesses as {param: value_array}.
        param_names: List of required parameter names to ensure correct ordering.
        image_shape: Shape of the image.
        pixel_indices: List of pixel indices for which to transform p0.
    Returns:
        np.ndarray: Initial parameter guesses as a numpy array in the correct order.
    """
    validate_parameter_names(p0, param_names)

    for param in param_names:
        if not isinstance(p0[param], np.ndarray):
            raise ValueError(f"p0 for parameter '{param}' must be a numpy array.")
        if p0[param].shape != image_shape:
            raise ValueError(
                f"p0 for parameter '{param}' must have shape {image_shape}."
            )

    p0_entries = []
    if pixel_indices is None:
        pixel_indices = list(np.ndindex(image_shape))

    for param in param_names:
        values = np.array(
            [p0[param][idx] for idx in pixel_indices]
        )  # Shape: (n_pixel,)
        p0_entries.append(values)

    p0_array = np.stack(p0_entries, axis=-1)  # Shape: (n_pixel, n_param)
    return p0_array


# transform bounds for curve_fit, ensuring they are in the correct format and shape
def transform_bounds(
    bounds: dict[str, tuple[float, float]],
    param_names: list[str],
    n_pixels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform bounds dictionary to numpy arrays (Not Spatial).

    Args:
        bounds (dict): Dictionary with parameter names as keys and (lower, upper) tuples as values.
        param_names (list): List of parameter names to ensure correct ordering.
        n_pixels (int | None): Number of pixels for which to transform bounds.
    Returns:
        tuple[np.ndarray, np.ndarray]: Lower and upper bounds as numpy arrays.
    """
    lower_bounds = []
    upper_bounds = []

    # Check for missing or extra parameters in bounds
    validate_parameter_names(bounds, param_names)

    # Transform float dictionary bounds to numpy arrays in the correct order
    if isinstance(bounds[param_names[0]][0], (int, float)) and isinstance(
        bounds[param_names[0]][1], (int, float)
    ):
        if n_pixels is None:
            for param in param_names:
                lower, upper = bounds[param]
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            return np.array(lower_bounds), np.array(upper_bounds)
        else:
            # Extract lower and upper in param_names order
            lower = np.array(
                [bounds[p][0] for p in param_names], dtype=float
            )  # Shape: (n_param,)
            upper = np.array(
                [bounds[p][1] for p in param_names], dtype=float
            )  # Shape: (n_param,)
            # Replicate across pixels: shape (n_param, n_pixel)
            lower_array = np.tile(
                lower[:, np.newaxis], (1, n_pixels)
            )  # Shape: (n_param, n_pixel)
            upper_array = np.tile(
                upper[:, np.newaxis], (1, n_pixels)
            )  # Shape: (n_param, n_pixel)
            return lower_array, upper_array
    else:
        raise ValueError(
            "Bounds must be provided as a dictionary with parameter names as keys and (lower, upper) tuples as values, where lower and upper are floats."
        )


def transform_bounds_spatial(
    bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    param_names: list[str],
    image_shape: tuple[int, int],
    pixel_indices: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform spatial bounds dictionary to separate dictionaries for lower and upper bounds.

    Args:
        bounds (dict): Dictionary with parameter names as keys and (lower_array, upper_array) tuples as values.
        param_names (list): List of parameter names to ensure correct ordering.
        image_shape (tuple[int, int]): Shape of the image.
        pixel_indices (list[tuple[int, int]] | None): List of pixel indices for which to transform bounds.
    Returns:
        tuple[np.ndarray, np.ndarray]: Lower and upper bounds as numpy arrays.
    """

    # Validate bounds and check for missing or extra parameters
    validate_parameter_names(bounds, param_names)

    # Validate that each bound is a tuple of two numpy arrays with the correct shape
    for param in param_names:
        lower, upper = bounds[param]
        if not (isinstance(lower, np.ndarray) and isinstance(upper, np.ndarray)):
            raise ValueError(f"Bounds for parameter '{param}' must be numpy arrays.")
        if lower.shape != image_shape or upper.shape != image_shape:
            raise ValueError(
                f"Bounds for parameter '{param}' must have shape {image_shape}."
            )

    # Initialize lists to store extracted values
    lower_entries = []
    upper_entries = []

    if pixel_indices is None:
        pixel_indices = list(np.ndindex(image_shape))

    # Iterate over the parameter dictionary
    for param, (lower_array, upper_array) in bounds.items():
        lower_values = np.array(
            [lower_array[idx] for idx in pixel_indices]
        )  # Shape: (n_pixel,)
        upper_values = np.array(
            [upper_array[idx] for idx in pixel_indices]
        )  # Shape: (n_pixel,)
        # Append the extracted values to the respective lists
        lower_entries.append(lower_values)
        upper_entries.append(upper_values)

    # Stack the lists to create final lower and upper bound arrays
    lower_array = np.stack(lower_entries, axis=-1)  # Shape: (n_pixel, n_param)
    upper_array = np.stack(upper_entries, axis=-1)  # Shape: (n_pixel, n_param)
    return lower_array, upper_array
