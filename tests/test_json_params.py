"""
Tests for the JSON parameter loading functionality.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyneapple.parameters.parameters import BaseParams
from pyneapple.utils.exceptions import ClassMismatch

from .test_toolbox import ParameterTools


class TestJsonParams:
    """Test cases for the JSON parameter loading functionality."""

    def test_load_json_params(self):
        """Test loading parameters from a JSON file."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            _ = f.write(
                """
                {
                    "General": {
                        "Class": "BaseParams",
                        "fit_type": "single",
                        "max_iter": 100,
                        "fit_tolerance": 1e-6,
                        "n_pools": 4
                    },
                    "Model": {
                        "model": "test-model",
                        "fit_reduced": false
                    },
                    "boundaries": {
                        "n_bins": 250
                    }
                }
                """
            )
            temp_file = f.name

        try:
            # Load the JSON file
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

    def test_load_from_json(self):
        """Test loading parameters using load_from_json method."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            _ = f.write(
                """
                {
                    "General": {
                        "Class": "BaseParams",
                        "fit_type": "multi",
                        "max_iter": 200
                    },
                    "Model": {}
                }
                """
            )
            temp_file = f.name

        try:
            # Create parameters and load from JSON
            params = BaseParams()
            params.load_from_json(temp_file)

            # Check the loaded parameters
            assert params.fit_type == "multi"
            assert params.max_iter == 200
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    def test_missing_class_identifier(self):
        """Test error when Class identifier is missing."""
        # Create a temporary JSON file without Class identifier
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            _ = f.write(
                """
                {
                    "General": {
                        "fit_type": "single",
                        "max_iter": 100
                    }
                }
                """
            )
            temp_file = f.name

        try:
            # Try to load the JSON file without Class identifier
            with pytest.raises(ClassMismatch):
                BaseParams(temp_file)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    def test_invalid_json(self):
        """Test error when JSON file is invalid."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            _ = f.write(
                """
                {
                    "General": {
                        "Class": "BaseParams",
                        "fit_type": "single",
                        "max_iter": 100,
                    }
                }
                """
            )
            temp_file = f.name

        try:
            # Try to load the invalid JSON file
            with pytest.raises(Exception):
                BaseParams(temp_file)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    def test_save_to_json(self):
        """Test saving parameters to a JSON file."""
        # Create a parameters object
        params = BaseParams()
        params.fit_type = "single"
        params.max_iter = 100
        params.fit_tolerance = 1e-6
        params.n_pools = 4
        params.b_values = np.array([0, 100, 200, 400, 800])

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        temp_path = Path(temp_file)

        try:
            # Test save/load roundtrip using helper
            ParameterTools.assert_save_load_roundtrip(
                params, temp_path, BaseParams, "save_to_json"
            )
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    def test_json_with_numpy_arrays(self):
        """Test that numpy arrays are properly serialized and deserialized."""
        params = BaseParams()
        params.b_values = np.array([0, 50, 100, 200, 400, 600, 800, 1000])
        params.fit_type = "single"

        temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        temp_path = Path(temp_file)

        try:
            # Save and load
            params.save_to_json(temp_path)
            loaded_params = BaseParams(temp_path)

            # Check that arrays are equal
            assert np.array_equal(
                loaded_params.b_values.squeeze(), params.b_values.squeeze()
            )
        finally:
            os.unlink(temp_file)

    def test_json_with_nested_structures(self):
        """Test JSON handling of nested dictionaries."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            _ = f.write(
                """
                {
                    "General": {
                        "Class": "BaseParams",
                        "fit_type": "single"
                    },
                    "Model": {},
                    "boundaries": {
                        "n_bins": 250,
                        "nested": {
                            "param1": 10,
                            "param2": 20
                        }
                    }
                }
                """
            )
            temp_file = f.name

        try:
            params = BaseParams(temp_file)
            assert params.boundaries["n_bins"] == 250
            assert params.boundaries["nested"]["param1"] == 10
            assert params.boundaries["nested"]["param2"] == 20
        finally:
            os.unlink(temp_file)

    def test_json_type_preservation(self):
        """Test that different data types are preserved correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            _ = f.write(
                """
                {
                    "General": {
                        "Class": "BaseParams",
                        "fit_type": "single",
                        "max_iter": 100,
                        "fit_tolerance": 0.000001
                    },
                    "Model": {
                        "fit_reduced": false
                    },
                    "boundaries": {
                        "use_bounds": true
                    }
                }
                """
            )
            temp_file = f.name

        try:
            params = BaseParams(temp_file)

            # Check types
            assert isinstance(params.fit_type, str)
            assert isinstance(params.max_iter, int)
            assert isinstance(params.fit_tolerance, float)
            assert isinstance(params.fit_model.fit_reduced, bool)
            assert isinstance(params.boundaries["use_bounds"], bool)
        finally:
            os.unlink(temp_file)

    @pytest.mark.parametrize(
        "extension,mode,content,expected_model",
        [
            (
                ".json",
                "w",
                '{"General": {"Class": "BaseParams"}, "Model": {"model": "json-model"}}',
                "json-model",
            ),
            (
                ".toml",
                "wb",
                b'[General]\nClass = "BaseParams"\n[Model]\nmodel = "toml-model"',
                "toml-model",
            ),
        ],
    )
    def test_extension_based_loading(self, extension, mode, content, expected_model):
        """Test that the correct loader is chosen based on file extension.
        
        Tests both JSON and TOML file loading to ensure the parameter loader
        correctly dispatches to the appropriate parser based on file extension.
        """
        # Create a temporary file with the specified extension
        with tempfile.NamedTemporaryFile(
            suffix=extension, mode=mode, delete=False
        ) as f:
            _ = f.write(content)
            temp_file = f.name

        try:
            # Load parameters from file
            params = BaseParams(temp_file)
            assert params.fit_model.name == expected_model
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)
