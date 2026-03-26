"""NIfTI file I/O operations for DWI data.

This module provides functions for loading and saving DWI data in NIfTI format,
including 4D volume handling, 2D slice extraction, and parameter map saving.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional
from loguru import logger


def load_dwi_nifti(path: str) -> tuple[np.ndarray, nib.Nifti1Image]:  # type: ignore
    """Load 4D DWI NIfTI file.

    Parameters
    ----------
    path : str
        Path to the NIfTI file (.nii or .nii.gz).

    Returns
    -------
    data : np.ndarray
        4D array with shape (x, y, slice, b_value).
    nifti_img : nibabel.Nifti1Image
        NIfTI image object containing affine and header information.

    Raises
    ------
    FileNotFoundError
        If the NIfTI file does not exist.
    ValueError
        If the file is not a valid NIfTI file or has unexpected dimensions.

    Examples
    --------
    >>> data, img = load_dwi_nifti('data/dwi.nii.gz')
    >>> print(data.shape)  # (64, 64, 30, 10) for example
    """
    path_obj = Path(path)

    # Check file exists
    if not path_obj.exists():
        logger.error(f"NIfTI file not found: {path}")
        raise FileNotFoundError(
            f"NIfTI file not found: {path}\n"
            f"Please check the file path and ensure the file exists."
        )

    # Try to load the NIfTI file
    try:
        nifti_img: nib.Nifti1Image = nib.load(str(path_obj))  # type: ignore
        data = nifti_img.get_fdata()
    except Exception as e:
        logger.error(f"Failed to load NIfTI file {path}: {e}")
        raise ValueError(
            f"Failed to load NIfTI file: {path}\n"
            f"Error: {e}\n"
            f"Ensure the file is a valid NIfTI format (.nii or .nii.gz)."
        )

    # Validate dimensions
    if data.ndim not in (3, 4):
        logger.error(f"Unexpected NIfTI dimensions: {data.ndim}D (expected 3D or 4D)")
        raise ValueError(
            f"Unexpected NIfTI dimensions: {data.ndim}D\n"
            f"Expected 3D (x, y, z) or 4D (x, y, z, b_value) data.\n"
            f"Got shape: {data.shape}"
        )

    # If 3D, assume single b-value and add dimension
    if data.ndim == 3:
        logger.warning(
            f"Loaded 3D NIfTI with shape {data.shape}, "
            f"adding b-value dimension to get 4D array"
        )
        data = data[..., np.newaxis]

    logger.debug(f"Successfully loaded NIfTI: {path}, shape: {data.shape}")

    return data, nifti_img


def extract_2d_slice(volume_4d: np.ndarray, slice_idx: int) -> np.ndarray:
    """Extract a 2D slice from a 4D DWI volume.

    Parameters
    ----------
    volume_4d : np.ndarray
        4D array with shape (x, y, n_slices, n_bvalues).
    slice_idx : int
        Index of the slice to extract (0-indexed).

    Returns
    -------
    slice_3d : np.ndarray
        3D array with shape (x, y, n_bvalues).

    Raises
    ------
    ValueError
        If volume_4d is not 4D or slice_idx is out of bounds.

    Examples
    --------
    >>> data_4d = np.random.rand(64, 64, 30, 10)
    >>> slice_2d = extract_2d_slice(data_4d, slice_idx=15)
    >>> print(slice_2d.shape)  # (64, 64, 10)
    """
    # Validate input is 4D
    if volume_4d.ndim != 4:
        logger.error(
            f"Expected 4D volume, got {volume_4d.ndim}D with shape {volume_4d.shape}"
        )
        raise ValueError(
            f"Expected 4D volume (x, y, slices, b_values), "
            f"got {volume_4d.ndim}D array with shape {volume_4d.shape}."
        )

    n_slices = volume_4d.shape[2]

    # Validate slice index
    if not (0 <= slice_idx < n_slices):
        logger.error(f"Slice index {slice_idx} out of range [0, {n_slices-1}]")
        raise ValueError(
            f"Slice index {slice_idx} is out of bounds.\n"
            f"Volume has {n_slices} slices (valid indices: 0 to {n_slices-1})."
        )

    # Extract slice
    slice_3d = volume_4d[:, :, slice_idx, :]

    logger.debug(
        f"Extracted slice {slice_idx}/{n_slices-1}, " f"shape: {slice_3d.shape}"
    )

    return slice_3d


def save_parameter_map(
    params: dict[str, np.ndarray],
    path: str,
    reference_nifti: nib.Nifti1Image,  # type: ignore
    param_name: Optional[str] = None,
) -> None:
    """Save parameter map(s) to NIfTI file, preserving affine transformation.

    Parameters
    ----------
    params : dict
        dictionary mapping parameter names to 2D arrays.
    path : str
        Output path for the NIfTI file (.nii or .nii.gz).
    reference_nifti : nibabel.Nifti1Image
        Reference NIfTI image to copy affine and header from.
    param_name : str, optional
        If provided, save only this parameter. Otherwise, save all parameters
        as separate volumes in a 3D/4D array.

    Raises
    ------
    ValueError
        If parameters have inconsistent shapes or param_name not found.

    Examples
    --------
    >>> params = {'S0': np.ones((64, 64)), 'D': np.ones((64, 64)) * 0.001}
    >>> _, ref_img = load_dwi_nifti('data/dwi.nii.gz')
    >>> save_parameter_map(params, 'results/S0_map.nii.gz', ref_img, param_name='S0')
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Select parameter(s) to save
    if param_name is not None:
        if param_name not in params:
            logger.error(f"Parameter '{param_name}' not found in params dict")
            raise ValueError(
                f"Parameter '{param_name}' not found.\n"
                f"Available parameters: {list(params.keys())}"
            )
        data_to_save = params[param_name]
        logger.debug(f"Saving single parameter: {param_name}")
    else:
        # Stack all parameters along new axis
        param_arrays = list(params.values())

        # Validate shapes are consistent
        shapes = [arr.shape for arr in param_arrays]
        if len(set(shapes)) > 1:
            logger.error(
                f"Inconsistent parameter shapes: {dict(zip(params.keys(), shapes))}"
            )
            raise ValueError(
                f"Parameter arrays have inconsistent shapes:\n"
                f"{dict(zip(params.keys(), shapes))}\n"
                f"All parameters must have the same spatial dimensions."
            )

        # Stack parameters
        if len(param_arrays) == 1:
            data_to_save = param_arrays[0]
        else:
            data_to_save = np.stack(param_arrays, axis=-1)

        logger.debug(f"Saving {len(param_arrays)} parameters: {list(params.keys())}")

    # Ensure at least 3D for NIfTI
    if data_to_save.ndim == 2:
        data_to_save = data_to_save[..., np.newaxis]

    # Create new NIfTI image with reference affine
    new_img = nib.Nifti1Image(  # type: ignore
        data_to_save, reference_nifti.affine, reference_nifti.header
    )

    # Save
    try:
        nib.save(new_img, str(path_obj))  # type: ignore
        logger.info(f"Saved parameter map to: {path}")
    except Exception as e:
        logger.error(f"Failed to save NIfTI file {path}: {e}")
        raise ValueError(f"Failed to save NIfTI file: {path}\nError: {e}")


def normalize_dwi(
    data: np.ndarray, b0_indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """Normalize DWI signal by b0 images.

    Parameters
    ----------
    data : np.ndarray
        DWI data array, last dimension should be b-values.
    b0_indices : np.ndarray, optional
        Indices of b0 images (b-value = 0). If None, assumes first volume is b0.

    Returns
    -------
    normalized : np.ndarray
        Normalized DWI data (signal / mean_b0).

    Examples
    --------
    >>> data = np.random.rand(64, 64, 10)
    >>> normalized = normalize_dwi(data, b0_indices=np.array([0]))
    """
    if b0_indices is None:
        # Assume first volume is b0
        b0_indices = np.array([0])
        logger.debug("No b0_indices provided, using first volume as b0")

    # Extract b0 images and compute mean
    b0_data = data[..., b0_indices]
    mean_b0 = np.mean(b0_data, axis=-1, keepdims=True)

    # Avoid division by zero
    mean_b0 = np.where(mean_b0 > 0, mean_b0, 1.0)

    # Normalize
    normalized = data / mean_b0

    logger.debug(f"Normalized DWI data using {len(b0_indices)} b0 image(s)")

    return normalized


def create_mask(data: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Create binary mask from DWI data.

    Uses mean signal across b-values to identify tissue.

    Parameters
    ----------
    data : np.ndarray
        DWI data array, last dimension should be b-values.
    threshold : float, optional
        Threshold as fraction of maximum mean signal (default: 0.1).

    Returns
    -------
    mask : np.ndarray
        Binary mask (1 = tissue, 0 = background), same spatial shape as input.

    Examples
    --------
    >>> data = np.random.rand(64, 64, 10)
    >>> mask = create_mask(data, threshold=0.1)
    >>> print(mask.shape)  # (64, 64)
    """
    # Compute mean signal across b-values
    mean_signal = np.mean(data, axis=-1)

    # Threshold
    threshold_value = threshold * np.max(mean_signal)
    mask = (mean_signal > threshold_value).astype(np.uint8)

    n_voxels = np.sum(mask)
    total_voxels = np.prod(mask.shape)
    logger.debug(
        f"Created mask: {n_voxels}/{total_voxels} voxels "
        f"({100*n_voxels/total_voxels:.1f}%)"
    )

    return mask
