"""Utility classes and functions for test suite.

This module provides helper tools for testing including:

- ParameterTools: Class for creating test parameter objects
- Data generators: Creating synthetic IVIM signals with known parameters
- Comparison utilities: Asserting parameter recovery accuracy
- Test fixtures: Reusable test data and configurations

These utilities support testing by providing standardized test data
generation and validation methods used across multiple test files.
"""
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from pyneapple import IVIMParams


class ParameterTools(object):
    @staticmethod
    def get_attributes(item) -> list:
        return [
            attr
            for attr in dir(item)
            if not callable(getattr(item, attr))
            and not attr.startswith("_")
            and not isinstance(getattr(item, attr), partial)
            and not attr in ["fit_model", "fit_function", "params_1", "params_2"]
        ]

    @staticmethod
    def compare_boundaries(boundary1, boundary2):
        attributes1 = ParameterTools.get_attributes(boundary1)
        attributes2 = ParameterTools.get_attributes(boundary2)

        assert attributes1 == attributes2

        for attr in attributes1:
            if isinstance(getattr(boundary1, attr), np.ndarray):
                assert getattr(boundary1, attr).all() == getattr(boundary2, attr).all()
            else:
                assert getattr(boundary1, attr) == getattr(boundary2, attr)

    @staticmethod
    def compare_parameters(params1, params2) -> list:
        """
        Compares two parameter sets.

        Args:
            params1: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams
            params2: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams

        Returns:

        """
        # compare attributes first
        attributes = ParameterTools.get_attributes(params1)
        test_attributes = ParameterTools.get_attributes(params2)

        # Atleast all original parameters should be present in the test parameters
        assert set(attributes).issubset(set(test_attributes))
        return attributes

    @staticmethod
    def compare_attributes(params1, params2, attributes: list):
        """
        Compares attribute values of two parameter sets
        Args:
            params1: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams
            params2: Parameters, IVIMParams, NNLSParams, NNLSCVParams, IDEALParams
            attributes:

        Returns:

        """
        for attr in attributes:
            if isinstance(getattr(params1, attr), np.ndarray):
                assert getattr(params1, attr).all() == getattr(params2, attr).all()
            elif attr.lower() == "boundaries":
                ParameterTools.compare_boundaries(
                    getattr(params1, attr), getattr(params2, attr)
                )
            elif attr in ["fit_model", "fit_function"]:
                continue
            elif attr in ["params_1", "params_2"]:  # Special case for SegmentedIVIM
                continue
            else:
                assert getattr(params1, attr) == getattr(params2, attr)

    @staticmethod
    def assert_save_load_roundtrip(params, file_path, param_class, save_method="save_to_json"):
        """Helper to test parameter save/load roundtrip.
        
        Tests that parameters can be saved and loaded without loss of information.
        This consolidates the repetitive save→load→compare pattern seen across
        multiple test files.
        
        Args:
            params: Parameter object to save (IVIMParams, NNLSParams, etc.)
            file_path: Path to save/load file
            param_class: Class to use for loading (e.g., IVIMParams, NNLSParams)
            save_method: Method name to call for saving ("save_to_json" or "save_to_toml")
        
        Example:
            >>> ParameterTools.assert_save_load_roundtrip(
            ...     ivim_params, out_json, IVIMParams, "save_to_json"
            ... )
        """
        # Save parameters
        getattr(params, save_method)(file_path)
        
        # Load parameters
        loaded_params = param_class(file_path)
        
        # Compare parameters
        attributes = ParameterTools.compare_parameters(params, loaded_params)
        ParameterTools.compare_attributes(params, loaded_params, attributes)

    @staticmethod
    def assert_pixel_args_structure(pixel_args, expected_tuple_length, img_shape=None):
        """Helper to validate structure of get_pixel_args output.
        
        Tests that pixel argument tuples have the expected structure and
        components. This consolidates repetitive pixel_args validation
        across parameter tests.
        
        Args:
            pixel_args: Output from params.get_pixel_args()
            expected_tuple_length: Expected number of elements in each tuple
                (2 for general boundaries: (coords, signal)
                 5 for individual boundaries: (coords, signal, x0, lb, ub))
            img_shape: Optional image shape to verify signal length
        
        Example:
            >>> args = params.get_pixel_args(img, seg)
            >>> ParameterTools.assert_pixel_args_structure(args, 2, img.shape)
        """
        args_list = list(pixel_args)
        
        # Check we have arguments
        assert len(args_list) > 0, "No pixel arguments generated"
        
        # Check each tuple structure
        for arg in args_list:
            assert len(arg) == expected_tuple_length, \
                f"Expected {expected_tuple_length} elements, got {len(arg)}"
            
            # First element should be coordinates (i, j, k)
            assert len(arg[0]) == 3, "Coordinates should be (i, j, k)"
            
            # Second element should be signal array
            if img_shape is not None:
                assert len(arg[1]) == img_shape[-1], \
                    f"Signal length {len(arg[1])} doesn't match image volumes {img_shape[-1]}"

    @staticmethod
    def assert_export_creates_files(results, export_method, file_path, expected_files, cleanup=True, **kwargs):
        """Helper to test results export methods create expected output files.
        
        Consolidates the repetitive save→verify→cleanup pattern seen across
        Excel, NIfTI, and HDF5 export tests.
        
        Args:
            results: Results object (BaseResults, IVIMResults, NNLSResults)
            export_method: Method name to call (e.g., "save_to_excel", "save_to_nii")
            file_path: Path to save file(s)
            expected_files: List of expected output file paths (Path objects or strings)
            cleanup: Whether to delete files after verification (default: True)
            **kwargs: Additional arguments to pass to export method
        
        Example:
            >>> ParameterTools.assert_export_creates_files(
            ...     results, "save_to_excel", out_excel, [out_excel],
            ...     split_index=True, is_segmentation=False
            ... )
        """
        # Call the export method
        getattr(results, export_method)(file_path, **kwargs)
        
        # Verify all expected files exist
        for expected_file in expected_files:
            expected_path = Path(expected_file)
            assert expected_path.exists(), f"Expected file not created: {expected_path}"
        
        # Cleanup if requested
        if cleanup:
            for expected_file in expected_files:
                expected_path = Path(expected_file)
                if expected_path.exists():
                    expected_path.unlink()

    @staticmethod
    def assert_export_file_content(results, export_method, file_path, loader_func, 
                                   content_validator, cleanup=True, **export_kwargs):
        """Helper to test results export with content validation.
        
        Extends assert_export_creates_files by also loading and validating
        the exported file content. Useful for verifying data integrity.
        
        Args:
            results: Results object to export
            export_method: Method name to call for export
            file_path: Path to save file
            loader_func: Function to load the exported file (e.g., pd.read_excel)
            content_validator: Callable that takes loaded content and performs assertions
            cleanup: Whether to delete file after validation
            **export_kwargs: Arguments to pass to export method
        
        Example:
            >>> def validate_excel(df):
            ...     assert "parameter" in df.columns
            ...     assert len(df) > 0
            >>> ParameterTools.assert_export_file_content(
            ...     results, "save_to_excel", out_excel, pd.read_excel,
            ...     validate_excel, split_index=True
            ... )
        """
        # Export the file
        getattr(results, export_method)(file_path, **export_kwargs)
        
        # Verify file exists
        assert Path(file_path).exists(), f"File not created: {file_path}"
        
        # Load and validate content
        loaded_content = loader_func(file_path)
        content_validator(loaded_content)
        
        # Cleanup if requested
        if cleanup:
            Path(file_path).unlink()

    @staticmethod
    def assert_fit_completes(fit_data, fit_method, **fit_kwargs):
        """Helper to test that fitting completes without errors.
        
        Consolidates the common pattern of calling a fit method and asserting
        it completes successfully. Used across IVIM, NNLS, and IDEAL tests.
        
        Args:
            fit_data: FitData object with image, segmentation, and parameters
            fit_method: Method name to call (e.g., "fit_pixel_wise", "fit_segmentation_wise")
            **fit_kwargs: Arguments to pass to fit method (e.g., fit_type="multi")
        
        Example:
            >>> ParameterTools.assert_fit_completes(
            ...     ivim_fit_data, "fit_pixel_wise", fit_type="multi"
            ... )
        """
        # Call the fitting method
        getattr(fit_data, fit_method)(**fit_kwargs)
        
        # Basic assertion - fitting completed without raising exceptions
        # Additional validations can check results object if needed
        assert fit_data.results is not None, "Results object should be populated after fitting"
