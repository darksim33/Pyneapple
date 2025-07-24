"""
Tests for the TOML parameter loading functionality.
"""

import os
import tempfile
from pathlib import Path

import pytest
import numpy as np

from pyneapple.parameters.parameters import BaseParams
from pyneapple.utils.exceptions import ClassMismatch


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
                number_points = 250
            """
            )
            temp_file = f.name

        try:
            # Load the TOML file
            params = BaseParams(temp_file)

            # Check the loaded parameters
            assert params.model == "test-model"
            assert params.fit_type == "single"
            assert params.fit_reduced is False
            assert params.max_iter == 100
            assert params.fit_tolerance == 1e-6
            assert params.n_pools == 4
            assert params.boundaries.number_points == 250
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
            f.write(
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
            f.write(
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
        params.model = "test-model"
        params.fit_type = "single"
        params.fit_reduced = False
        params.max_iter = 100
        params.fit_tolerance = 1e-6
        params.n_pools = 4
        params.b_values = np.array([0, 100, 200, 400, 800])

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".toml", delete=False).name
        temp_path = Path(temp_file)

        try:
            # Save parameters to TOML
            params.save_to_toml(temp_path)

            # Load parameters from saved TOML
            loaded_params = BaseParams(temp_path)

            # Check the loaded parameters
            assert loaded_params.model == "test-model"
            assert loaded_params.fit_type == "single"
            assert loaded_params.fit_reduced is False
            assert loaded_params.max_iter == 100
            assert loaded_params.fit_tolerance == 1e-6
            assert loaded_params.n_pools == 4
            assert np.array_equal(
                loaded_params.b_values.squeeze(), np.array([0, 100, 200, 400, 800])
            )
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    def test_extension_based_loading(self):
        """Test that the correct loader is chosen based on file extension."""
        # Create a temporary TOML file
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb", delete=False) as f:
            f.write(
                b"""
                    [General]
                    Class = "BaseParams"
                    [Model]
                    model = "toml-model"
                """
            )
            toml_file = f.name

        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write(
                """
                    {
                        "General": {
                            "Class": "BaseParams"
                        },
                        "Model": {
                            "model": "json-model"
                        }
                    }
                """
            )
            json_file = f.name

        try:
            # Load parameters from TOML file
            toml_params = BaseParams(toml_file)
            assert toml_params.model == "toml-model"

            # Load parameters from JSON file
            json_params = BaseParams(json_file)
            assert json_params.model == "json-model"
        finally:
            # Clean up the temporary files
            os.unlink(toml_file)
            os.unlink(json_file)
