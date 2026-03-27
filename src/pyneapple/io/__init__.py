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
    reconstruct_maps,
    save_spectrum_to_nifti,
)
from .bvalue import (
    load_bvalues,
    save_bvalues,
)
from .toml import load_config, FittingConfig
from .hdf5 import save_to_hdf5, load_from_hdf5, save_params_to_hdf5
from .excel import save_params_to_excel, save_spectrum_to_excel


__all__ = [
    # NIfTI utilities
    "load_dwi_nifti",
    "extract_2d_slice",
    "save_parameter_map",
    "normalize_dwi",
    "create_mask",
    "reconstruct_maps",
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
    # Excel utilities
    "save_params_to_excel",
    "save_spectrum_to_excel",
]
