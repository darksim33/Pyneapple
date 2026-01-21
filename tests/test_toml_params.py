"""
Tests for the TOML parameter loading functionality.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyneapple.parameters.parameters import BaseParams
from pyneapple.utils.exceptions import ClassMismatch

from .test_toolbox import ParameterTools


class TestTomlParams:
    """Test cases for the TOML parameter loading functionality."""

    def test_load_toml_params(self):
        """Test loading parameters from a TOML file."""
        # Create a temporary TOML file
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb", delete=False) as f:
            f.write(
                b"""
                # Test TOML Parameter File
                [General]
                Class = "BaseParams"

                # Basic parameters
                fit_type = "single"
                max_iter = 100
                fit_tolerance = 1e-6
                n_pools = 4

                [Model]
                model = "test-model"
                fit_reduced = false

                # Boundaries section
                [boundaries]
                n_bins = 250
            """
            )
            temp_file = f.name

        try:
            # Load the TOML file
            params = BaseParams(temp_file)

            # Check the loaded parameters
            assert params.fit_type == "single"
            assert params.max_iter == 100
            assert params.fit_tolerance == 1e-6
            assert params.n_pools == 4
            assert params.fit_model.name == "test-model"
            assert params.fit_model.fit_reduced is False
            assert params.boundaries["n_bins"] == 250
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    def test_load_from_toml(self):
        """Test loading parameters using load_from_toml method."""
        # Create a temporary TOML file
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb", delete=False) as f:
            f.write(
                b"""
                # Test TOML Parameter File
                [General]
                Class = "BaseParams"
                fit_type = "multi"
                max_iter = 200
                [Model]
            """
            )
            temp_file = f.name

        try:
            # Create parameters and load from TOML
            params = BaseParams()
            params.load_from_toml(temp_file)

            # Check the loaded parameters
            assert params.fit_type == "multi"
            assert params.max_iter == 200
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    def test_missing_class_identifier(self):
        """Test error when Class identifier is missing."""
        # Create a temporary TOML file without Class identifier
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb", delete=False) as f:
            _ = f.write(
                b"""
                    # Test TOML Parameter File without Class
                    [General]
                    fit_type = "single"
                    max_iter = 100
                """
            )
            temp_file = f.name

        try:
            # Try to load the TOML file without Class identifier
            with pytest.raises(ClassMismatch):
                BaseParams(temp_file)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    def test_invalid_toml(self):
        """Test error when TOML file is invalid."""
        # Create a temporary file with invalid TOML
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb", delete=False) as f:
            _ = f.write(
                b"""
                    # Invalid TOML
                    Class = "BaseParams"
                    fit_type = "single" max_iter = 100  # Missing line break
                """
            )
            temp_file = f.name

        try:
            # Try to load the invalid TOML file
            with pytest.raises(Exception):
                BaseParams(temp_file)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    def test_save_to_toml(self):
        """Test saving parameters to a TOML file."""
        # Create a parameters object
        params = BaseParams()
        params.fit_type = "single"
        params.max_iter = 100
        params.fit_tolerance = 1e-6
        params.n_pools = 4
        params.b_values = np.array([0, 100, 200, 400, 800])

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".toml", delete=False).name
        temp_path = Path(temp_file)

        try:
            # Test save/load roundtrip using helper
            ParameterTools.assert_save_load_roundtrip(
                params, temp_path, BaseParams, "save_to_toml"
            )
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)
