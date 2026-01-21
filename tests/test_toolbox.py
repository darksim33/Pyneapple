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
