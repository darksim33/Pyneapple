"""
Input/output utilities for medical imaging data.

This module provides functions for loading and saving NIfTI files,
parsing b-value files, and preprocessing DWI data.
"""

from .nifti import (
    load_dwi_nifti,
    extract_2d_slice,
    save_parameter_map,
    normalize_dwi,
    create_mask,
)
from .bvalue import (
    load_bvalues,
    save_bvalues,
)
from .toml import load_config, FittingConfig
from .hdf5 import save_to_hdf5, load_from_hdf5

__all__ = [
    "load_dwi_nifti",
    "extract_2d_slice",
    "save_parameter_map",
    "normalize_dwi",
    "create_mask",
    "load_bvalues",
    "save_bvalues",
    "load_config",
    "FittingConfig",
    "save_to_hdf5",
    "load_from_hdf5",
]
