"""
Input/output utilities for medical imaging data.

This module provides functions for loading and saving NIfTI files,
parsing b-value files, and preprocessing DWI data.
"""

from __future__ import annotations

from .bvalue import (
    load_bvalues,
    save_bvalues,
)
from .excel import save_params_to_excel, save_spectrum_to_excel
from .hdf5 import load_from_hdf5, save_params_to_hdf5, save_result_to_hdf5, save_to_hdf5
from .nifti import (
    create_mask,
    extract_2d_slice,
    load_dwi_nifti,
    normalize_dwi,
    reconstruct_maps,
    reconstruct_segmentation_maps,
    save_parameter_map,
    save_spectrum_to_nifti,
)
from .toml import FittingConfig, load_config

__all__ = [
    # NIfTI utilities
    "load_dwi_nifti",
    "extract_2d_slice",
    "save_parameter_map",
    "normalize_dwi",
    "create_mask",
    "reconstruct_maps",
    "reconstruct_segmentation_maps",
    "save_spectrum_to_nifti",
    # B-value utilities
    "load_bvalues",
    "save_bvalues",
    # TOML configuration
    "load_config",
    "FittingConfig",
    # HDF5 utilities
    "save_to_hdf5",
    "load_from_hdf5",
    "save_params_to_hdf5",
    "save_result_to_hdf5",
    # Excel utilities
    "save_params_to_excel",
    "save_spectrum_to_excel",
]
