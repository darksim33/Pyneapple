"""Comprehensive tests for HDF5 encoding and decoding functionality.

This test module covers:
- Basic data types (strings, integers, floats, booleans)
- Numpy arrays with sparse matrix compression
- Path objects
- Lists (distinguished from arrays)
- Nested dictionaries
- Special key types (int, tuple)
- Edge cases and error conditions
- Real-world usage scenarios

Note: Some tests may initially fail due to bugs in the current implementation:
1. dict_to_hdf5: Line iterates over dict keys only instead of items()
2. _encode_list: Uses built-in 'list' instead of parameter '_list'
3. _decode_path: Missing proper return statement for non-bytes case
4. _decode_list: Missing return statement
5. dict_to_hdf5: Always creates dataset even for special-encoded types
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from pyneapple.io.hdf5 import (
    load_from_hdf5,
    save_to_hdf5,
)


class TestBasicDataTypes:
    """Test encoding/decoding of basic Python data types."""

    def test_string(self, hdf5_file):
        """Test string encoding and decoding."""
        data = {"key": "value"}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert loaded["key"] == data["key"]

    def test_integer_value(self, hdf5_file):
        """Test integer value encoding and decoding."""
        data = {"int_key": 42}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert loaded["int_key"] == data["int_key"]

    def test_float_value(self, hdf5_file):
        """Test float value encoding and decoding."""
        data = {"float_key": 3.14159}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert np.isclose(loaded["float_key"], data["float_key"])

    def test_multiple_types(self, hdf5_file):
        """Test mixed data types in one dictionary."""
        data = {
            "string": "hello",
            "integer": 123,
            "float": 45.67,
            "boolean": True,
        }

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert loaded["string"] == data["string"]
        assert loaded["integer"] == data["integer"]
        assert np.isclose(loaded["float"], data["float"])
        assert loaded["boolean"] == data["boolean"]


class TestNumpyArrays:
    """Test numpy array encoding with sparse matrix compression."""

    def test_1d_array(self, hdf5_file):
        """Test 1D numpy array encoding and decoding."""
        data = {"array": np.array([1, 2, 3, 4, 5])}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["array"], np.ndarray)
        assert np.array_equal(loaded["array"], data["array"])

    def test_2d_array(self, hdf5_file):
        """Test 2D numpy array encoding and decoding."""
        data = {"matrix": np.array([[1, 2, 3], [4, 5, 6]])}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["matrix"], np.ndarray)
        assert np.array_equal(loaded["matrix"], data["matrix"])

    def test_3d_array(self, hdf5_file):
        """Test 3D numpy array encoding and decoding."""
        data = {"array_3d": np.random.rand(5, 5, 5)}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["array_3d"], np.ndarray)
        assert loaded["array_3d"].shape == data["array_3d"].shape
        assert np.allclose(loaded["array_3d"], data["array_3d"])

    def test_4d_array(self, hdf5_file):
        """Test 4D numpy array encoding and decoding."""
        data = {"array_4d": np.random.rand(5, 5, 5, 5)}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["array_4d"], np.ndarray)
        assert loaded["array_4d"].shape == data["array_4d"].shape
        assert np.allclose(loaded["array_4d"], data["array_4d"])

    def test_sparse_array(self, hdf5_file):
        """Test sparse array (mostly zeros) encoding and decoding."""
        sparse_data = np.zeros((100, 100))
        sparse_data[10, 10] = 1.0
        sparse_data[50, 50] = 2.0
        sparse_data[90, 90] = 3.0

        data = {"sparse": sparse_data}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["sparse"], np.ndarray)
        assert np.array_equal(loaded["sparse"], data["sparse"])

    def test_float_array(self, hdf5_file):
        """Test float array encoding and decoding."""
        data = {"floats": np.random.rand(10, 10)}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["floats"], np.ndarray)
        assert np.allclose(loaded["floats"], data["floats"])

    def test_empty_array(self, hdf5_file):
        """Test empty array encoding and decoding."""
        data = {"empty": np.array([])}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["empty"], np.ndarray)
        assert len(loaded["empty"]) == 0


class TestPathObjects:
    """Test Path object encoding and decoding."""

    def test_simple_path(self, hdf5_file):
        """Test simple path encoding and decoding."""
        data = {"path": Path("/some/test/path")}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["path"], Path)
        assert loaded["path"] == data["path"]

    def test_relative_path(self, hdf5_file):
        """Test relative path encoding and decoding."""
        data = {"rel_path": Path("relative/path/to/file.txt")}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["rel_path"], Path)
        assert loaded["rel_path"] == data["rel_path"]

    def test_windows_path(self, hdf5_file):
        """Test Windows-style path encoding and decoding."""
        data = {"win_path": Path("C:/Users/Test/Documents/file.txt")}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["win_path"], Path)
        assert loaded["win_path"] == data["win_path"]


class TestLists:
    """Test list encoding to distinguish from numpy arrays."""

    def test_simple_list(self, hdf5_file):
        """Test simple list encoding and decoding."""
        data = {"list": [1, 2, 3, 4, 5]}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["list"], list)
        assert loaded["list"] == data["list"]

    def test_list_vs_array(self, hdf5_file):
        """Test that lists and arrays are preserved correctly."""
        data = {"list": [1, 2, 3], "array": np.array([1, 2, 3])}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["list"], list)
        assert isinstance(loaded["array"], np.ndarray)

    def test_float_list(self, hdf5_file):
        """Test list of floats encoding and decoding."""
        data = {"float_list": [1.1, 2.2, 3.3, 4.4]}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert isinstance(loaded["float_list"], list)
        assert np.allclose(loaded["float_list"], data["float_list"])


class TestNestedDictionaries:
    """Test nested dictionary structures."""

    def test_simple_nested(self, hdf5_file):
        """Test simple nested dictionary."""
        data = {"level1": {"level2": {"value": 42}}}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert loaded["level1"]["level2"]["value"] == 42

    def test_complex_nested(self, hdf5_file):
        """Test complex nested dictionary with mixed types."""
        data = {
            "metadata": {
                "name": "test",
                "version": 1,
                "params": {"threshold": 0.5, "iterations": 100},
            },
            "data": {"array": np.random.rand(5, 5), "path": Path("/test/path")},
        }

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert loaded["metadata"]["name"] == data["metadata"]["name"]
        assert loaded["metadata"]["version"] == data["metadata"]["version"]
        assert (
            loaded["metadata"]["params"]["threshold"]
            == data["metadata"]["params"]["threshold"]
        )
        assert np.allclose(loaded["data"]["array"], data["data"]["array"])
        assert loaded["data"]["path"] == data["data"]["path"]


class TestSpecialKeys:
    """Test special key types (int, tuple)."""

    def test_integer_key(self, hdf5_file):
        """Test integer keys are preserved."""
        data = {42: "value_for_42", 100: "value_for_100"}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert 42 in loaded
        assert 100 in loaded
        assert loaded[42] == data[42]
        assert loaded[100] == data[100]

    def test_tuple_key(self, hdf5_file):
        """Test tuple keys are preserved."""
        data = {(0, 0): "origin", (1, 2): "point_1_2", (5, 10, 15): "3d_point"}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert (0, 0) in loaded
        assert (1, 2) in loaded
        assert (5, 10, 15) in loaded
        assert loaded[(0, 0)] == data[(0, 0)]
        assert loaded[(1, 2)] == data[(1, 2)]
        assert loaded[(5, 10, 15)] == data[(5, 10, 15)]

    def test_mixed_keys(self, hdf5_file):
        """Test mixed key types in one dictionary."""
        data = {"string_key": "string_value", 42: "int_value", (1, 2): "tuple_value"}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert "string_key" in loaded
        assert 42 in loaded
        assert (1, 2) in loaded


class TestEdgeCases:
    """Test edge cases and potential error conditions."""

    def test_empty_dict(self, hdf5_file):
        """Test empty dictionary encoding and decoding."""
        data = {}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert loaded == data

    def test_nested_empty_dicts(self, hdf5_file):
        """Test nested empty dictionaries."""
        data = {"outer": {"inner": {}}}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert "outer" in loaded
        assert "inner" in loaded["outer"]
        assert loaded["outer"]["inner"] == {}

    def test_unicode_strings(self, hdf5_file):
        """Test unicode string encoding and decoding."""
        data = {"unicode": "Hello 世界 🌍", "emoji": "🎉🎊✨"}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert loaded["unicode"] == data["unicode"]
        assert loaded["emoji"] == data["emoji"]

    def test_very_large_array(self, hdf5_file):
        """Test encoding of large arrays (sparse compression benefit)."""
        large_array = np.zeros((1000, 1000))
        # Add some non-zero values
        large_array[::100, ::100] = np.random.rand(10, 10)

        data = {"large": large_array}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert np.array_equal(loaded["large"], data["large"])


class TestCompressionOptions:
    """Test compression options for numpy arrays."""

    def test_default_compression(self, hdf5_file):
        """Test default compression settings."""
        data = {"array": np.random.rand(100, 100)}

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        assert np.allclose(loaded["array"], data["array"])
        with h5py.File(hdf5_file, "r") as f:
            dataset = f["array"]["data"]
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 4

    def test_lzf_compression(self, hdf5_file):
        """Test default compression settings."""
        data = {"array": np.random.rand(100, 100)}

        save_to_hdf5(data, hdf5_file, compression="lzf")
        loaded = load_from_hdf5(hdf5_file)

        assert np.allclose(loaded["array"], data["array"])

        with h5py.File(hdf5_file, "r") as f:
            dataset = f["array"]["data"]
            assert dataset.compression == "lzf"

    def test_max_compression(self, hdf5_file):
        """Test default compression settings."""
        data = {"array": np.random.rand(100, 100)}

        save_to_hdf5(data, hdf5_file, compression_opts=9)
        loaded = load_from_hdf5(hdf5_file)

        assert np.allclose(loaded["array"], data["array"])

        with h5py.File(hdf5_file, "r") as f:
            dataset = f["array"]["data"]
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 9


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_experiment_data(self, hdf5_file):
        """Test typical experiment data structure."""
        data = {
            "metadata": {
                "experiment_name": "Test Experiment",
                "date": "2024-01-01",
                "version": 1,
            },
            "config": {
                "output_path": Path("/output/results"),
                "parameters": {"threshold": 0.5, "iterations": 1000, "tolerance": 1e-6},
            },
            "results": {
                "measurements": np.random.rand(100, 50),
                "timestamps": [0.1, 0.2, 0.3, 0.4],
                "labels": ["A", "B", "C", "D"],
            },
            "voxel_data": {
                (0, 0, 0): np.array([1.0, 2.0, 3.0]),
                (1, 1, 1): np.array([4.0, 5.0, 6.0]),
            },
        }

        save_to_hdf5(data, hdf5_file)
        loaded = load_from_hdf5(hdf5_file)

        # Verify metadata
        assert (
            loaded["metadata"]["experiment_name"] == data["metadata"]["experiment_name"]
        )
        assert loaded["metadata"]["version"] == data["metadata"]["version"]

        # Verify config
        assert loaded["config"]["output_path"] == data["config"]["output_path"]
        assert (
            loaded["config"]["parameters"]["threshold"]
            == data["config"]["parameters"]["threshold"]
        )

        # Verify results
        assert np.allclose(
            loaded["results"]["measurements"], data["results"]["measurements"]
        )
        assert loaded["results"]["timestamps"] == data["results"]["timestamps"]

        # Verify voxel data
        assert (0, 0, 0) in loaded["voxel_data"]
        assert np.allclose(
            loaded["voxel_data"][(0, 0, 0)], data["voxel_data"][(0, 0, 0)]
        )

    def test_multiple_save_load_cycles(self, hdf5_file):
        """Test data integrity across multiple save/load cycles."""
        original_data = {
            "array": np.random.rand(20, 20),
            "path": Path("/test/path"),
            "nested": {"value": 42, "list": [1, 2, 3]},
        }

        # Cycle 1
        save_to_hdf5(original_data, hdf5_file)
        loaded1 = load_from_hdf5(hdf5_file)

        # Cycle 2
        save_to_hdf5(loaded1, hdf5_file)
        loaded2 = load_from_hdf5(hdf5_file)

        # Verify data is still correct
        assert np.allclose(loaded2["array"], original_data["array"])
        assert loaded2["path"] == original_data["path"]
        assert loaded2["nested"]["value"] == original_data["nested"]["value"]
        assert loaded2["nested"]["list"] == original_data["nested"]["list"]
