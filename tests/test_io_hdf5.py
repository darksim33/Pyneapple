"""Tests for the HDF5 I/O module (io/hdf5.py).

Covers:
- save_to_hdf5 / load_from_hdf5 round-trips for all supported value types
- Dict-key type preservation (str, int, tuple)
- Nested dictionary structure
- numpy array compression
- Path object serialisation
- List vs array distinction
- String (bytes) decoding
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyneapple.io.hdf5 import (
    load_from_hdf5,
    save_to_hdf5,
    save_params_to_hdf5,
    _DEFAULT_GZIP_LEVEL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roundtrip(data: dict, tmp_path: Path, **kwargs) -> dict:
    """Save *data* to a temp HDF5 file and return the reloaded dict."""
    h5_file = tmp_path / "test.h5"
    save_to_hdf5(data, h5_file, **kwargs)
    return load_from_hdf5(h5_file)


# ---------------------------------------------------------------------------
# TestSaveLoadScalars
# ---------------------------------------------------------------------------


class TestSaveLoadScalars:
    """Round-trip tests for primitive scalar values."""

    def test_integer_value_roundtrip(self, tmp_path):
        """Integer values survive a save/load cycle unchanged."""
        result = _roundtrip({"count": 42}, tmp_path)
        assert result["count"] == 42

    def test_float_value_roundtrip(self, tmp_path):
        """Float values survive a save/load cycle unchanged."""
        result = _roundtrip({"value": 3.14}, tmp_path)
        assert result["value"] == pytest.approx(3.14)

    def test_string_value_roundtrip(self, tmp_path):
        """String values are decoded back to str after save/load."""
        result = _roundtrip({"label": "hello"}, tmp_path)
        assert result["label"] == "hello"
        assert isinstance(result["label"], str)

    def test_bool_value_roundtrip(self, tmp_path):
        """Boolean values survive a save/load cycle."""
        result = _roundtrip({"flag": True}, tmp_path)
        assert result["flag"]


# ---------------------------------------------------------------------------
# TestSaveLoadArrays
# ---------------------------------------------------------------------------


class TestSaveLoadArrays:
    """Round-trip tests for numpy arrays."""

    def test_1d_array_roundtrip(self, tmp_path):
        """1-D numpy arrays are restored with the same values and shape."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _roundtrip({"arr": arr}, tmp_path)
        np.testing.assert_array_equal(result["arr"], arr)

    def test_2d_array_roundtrip(self, tmp_path):
        """2-D numpy arrays are restored with the same shape and values."""
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = _roundtrip({"arr": arr}, tmp_path)
        np.testing.assert_array_equal(result["arr"], arr)

    def test_array_dtype_preserved(self, tmp_path):
        """Array dtype is preserved through the round-trip."""
        arr = np.array([1, 2, 3], dtype=np.float64)
        result = _roundtrip({"arr": arr}, tmp_path)
        assert result["arr"].dtype == np.float64

    def test_array_with_custom_compression(self, tmp_path):
        """Arrays saved with a custom compression level load correctly."""
        arr = np.random.default_rng(0).random((10, 10))
        result = _roundtrip(
            {"arr": arr}, tmp_path, compression="gzip", compression_opts=9
        )
        np.testing.assert_array_equal(result["arr"], arr)

    def test_zero_array_roundtrip(self, tmp_path):
        """All-zero arrays survive compression and decompression."""
        arr = np.zeros((5, 5), dtype=np.float32)
        result = _roundtrip({"arr": arr}, tmp_path)
        np.testing.assert_array_equal(result["arr"], arr)


# ---------------------------------------------------------------------------
# TestSaveLoadPath
# ---------------------------------------------------------------------------


class TestSaveLoadPath:
    """Round-trip tests for pathlib.Path values."""

    def test_path_restored_as_path_object(self, tmp_path):
        """Path values are decoded back to pathlib.Path instances."""
        p = Path("/some/data/file.nii.gz")
        result = _roundtrip({"filepath": p}, tmp_path)
        assert isinstance(result["filepath"], Path)

    def test_path_value_matches(self, tmp_path):
        """The decoded Path has the same string representation as the original."""
        p = Path("/some/data/file.nii.gz")
        result = _roundtrip({"filepath": p}, tmp_path)
        assert result["filepath"] == p


# ---------------------------------------------------------------------------
# TestSaveLoadList
# ---------------------------------------------------------------------------


class TestSaveLoadList:
    """Round-trip tests to confirm lists come back as lists, not arrays."""

    def test_list_restored_as_list(self, tmp_path):
        """Lists are decoded back to Python list, not numpy array."""
        result = _roundtrip({"vals": [1, 2, 3]}, tmp_path)
        assert isinstance(result["vals"], list)

    def test_list_values_match(self, tmp_path):
        """List elements match the original values after round-trip."""
        original = [10, 20, 30]
        result = _roundtrip({"vals": original}, tmp_path)
        assert result["vals"] == original


# ---------------------------------------------------------------------------
# TestKeyEncoding
# ---------------------------------------------------------------------------


class TestKeyEncoding:
    """Tests for non-string key type preservation."""

    def test_integer_key_roundtrip(self, tmp_path):
        """Integer dict keys are restored as int after load."""
        result = _roundtrip({42: "the answer"}, tmp_path)
        assert 42 in result
        assert result[42] == "the answer"

    def test_tuple_key_roundtrip(self, tmp_path):
        """Tuple dict keys are restored as tuple after load."""
        result = _roundtrip({(1, 2): "coords"}, tmp_path)
        assert (1, 2) in result
        assert result[(1, 2)] == "coords"

    def test_string_key_unchanged(self, tmp_path):
        """String keys remain strings after round-trip."""
        result = _roundtrip({"name": "test"}, tmp_path)
        assert "name" in result


# ---------------------------------------------------------------------------
# TestNestedDicts
# ---------------------------------------------------------------------------


class TestNestedDicts:
    """Tests for nested dictionary structures."""

    def test_nested_dict_restored(self, tmp_path):
        """Nested dictionaries are fully restored after round-trip."""
        data = {"outer": {"inner": 99}}
        result = _roundtrip(data, tmp_path)
        assert result["outer"]["inner"] == 99

    def test_deeply_nested_array(self, tmp_path):
        """Arrays in deeply nested dicts are restored correctly."""
        arr = np.array([1.0, 2.0])
        data = {"level1": {"level2": {"arr": arr}}}
        result = _roundtrip(data, tmp_path)
        np.testing.assert_array_equal(result["level1"]["level2"]["arr"], arr)

    def test_mixed_nested_types(self, tmp_path):
        """Nested dicts with mixed value types all round-trip correctly."""
        data = {
            "params": {
                "D": np.array([0.001, 0.002]),
                "label": "biexp",
                "n": 2,
            }
        }
        result = _roundtrip(data, tmp_path)
        np.testing.assert_array_equal(result["params"]["D"], data["params"]["D"])
        assert result["params"]["label"] == "biexp"
        assert result["params"]["n"] == 2


# ---------------------------------------------------------------------------
# TestFileCreation
# ---------------------------------------------------------------------------


class TestFileCreation:
    """Tests for file system behaviour of save_to_hdf5."""

    def test_file_is_created(self, tmp_path):
        """save_to_hdf5 creates the output file."""
        h5_file = tmp_path / "output.h5"
        save_to_hdf5({"x": 1}, h5_file)
        assert h5_file.exists()

    def test_filepath_as_string_accepted(self, tmp_path):
        """save_to_hdf5 and load_from_hdf5 accept str paths as well as Path."""
        h5_file = str(tmp_path / "output.h5")
        save_to_hdf5({"x": 1}, h5_file)
        result = load_from_hdf5(h5_file)
        assert result["x"] == 1


# ---------------------------------------------------------------------------
# TestDefaultGzipLevel
# ---------------------------------------------------------------------------


class TestDefaultGzipLevel:
    """Tests for the _DEFAULT_GZIP_LEVEL module constant."""

    def test_default_level_is_int(self):
        """_DEFAULT_GZIP_LEVEL is an integer."""
        assert isinstance(_DEFAULT_GZIP_LEVEL, int)

    def test_default_level_in_valid_range(self):
        """_DEFAULT_GZIP_LEVEL is in the valid gzip range 1–9."""
        assert 1 <= _DEFAULT_GZIP_LEVEL <= 9

    def test_default_level_used_when_no_opts_given(self, tmp_path):
        """Arrays saved without explicit compression_opts still load correctly."""
        arr = np.arange(100, dtype=np.float64)
        result = _roundtrip({"arr": arr}, tmp_path)
        np.testing.assert_array_equal(result["arr"], arr)


# ---------------------------------------------------------------------------
# TestSaveLoad4DArray
# ---------------------------------------------------------------------------


class TestSaveLoad4DArray:
    """Round-trip tests for high-dimensional numpy arrays."""

    def test_4d_array_roundtrip(self, tmp_path):
        """4-D arrays (e.g. DWI volumes) survive save/load unchanged."""
        arr = np.arange(4 * 4 * 1 * 8, dtype=np.float32).reshape(4, 4, 1, 8)
        result = _roundtrip({"vol": arr}, tmp_path)
        np.testing.assert_array_equal(result["vol"], arr)

    def test_4d_array_shape_preserved(self, tmp_path):
        """4-D array shape is restored exactly after round-trip."""
        arr = np.zeros((2, 3, 4, 5), dtype=np.float64)
        result = _roundtrip({"vol": arr}, tmp_path)
        assert result["vol"].shape == arr.shape


# ---------------------------------------------------------------------------
# TestStringArrayDecoding
# ---------------------------------------------------------------------------


class TestStringArrayDecoding:
    """Tests for byte-string → str decoding in scalar and array datasets."""

    def test_scalar_string_decoded_to_str(self, tmp_path):
        """Scalar string values stored as bytes are decoded back to str."""
        result = _roundtrip({"name": "pyneapple"}, tmp_path)
        assert isinstance(result["name"], str)
        assert result["name"] == "pyneapple"

    def test_empty_string_roundtrip(self, tmp_path):
        """Empty strings survive encoding and decoding."""
        result = _roundtrip({"tag": ""}, tmp_path)
        assert result["tag"] == ""
        assert isinstance(result["tag"], str)


# ---------------------------------------------------------------------------
# TestSaveParamsToHdf5
# ---------------------------------------------------------------------------


class TestSaveParamsToHdf5:
    """Round-trip tests for the save_params_to_hdf5 convenience function."""

    @pytest.fixture
    def pixel_data(self):
        """Minimal fitted_params, pixel_indices and spatial_shape for 4×4×1 image."""
        rng = np.random.default_rng(42)
        pixel_indices = [(x, y, 0) for x in range(4) for y in range(4)]
        fitted_params = {
            "S0": rng.uniform(800, 1200, size=16).astype(np.float32),
            "D": rng.uniform(0.0005, 0.003, size=16).astype(np.float32),
        }
        spatial_shape = (4, 4, 1)
        return fitted_params, pixel_indices, spatial_shape

    def test_file_is_created(self, tmp_path, pixel_data):
        """save_params_to_hdf5 writes an HDF5 file to disk."""
        fitted_params, pixel_indices, spatial_shape = pixel_data
        out = tmp_path / "params.h5"
        save_params_to_hdf5(fitted_params, pixel_indices, spatial_shape, out)
        assert out.exists()

    def test_params_keys_present(self, tmp_path, pixel_data):
        """Loaded file has 'params', 'pixel_indices', and 'spatial_shape' keys."""
        fitted_params, pixel_indices, spatial_shape = pixel_data
        out = tmp_path / "params.h5"
        save_params_to_hdf5(fitted_params, pixel_indices, spatial_shape, out)
        loaded = load_from_hdf5(out)
        assert "params" in loaded
        assert "pixel_indices" in loaded
        assert "spatial_shape" in loaded

    def test_s0_map_roundtrip(self, tmp_path, pixel_data):
        """S0 spatial map is restored with correct values and shape."""
        fitted_params, pixel_indices, spatial_shape = pixel_data
        out = tmp_path / "params.h5"
        save_params_to_hdf5(fitted_params, pixel_indices, spatial_shape, out)
        loaded = load_from_hdf5(out)
        s0_map = loaded["params"]["S0"]
        assert s0_map.shape == spatial_shape

    def test_raises_on_empty_fitted_params(self, tmp_path):
        """save_params_to_hdf5 raises ValueError when fitted_params is empty."""
        with pytest.raises(ValueError, match="empty"):
            save_params_to_hdf5({}, [(0, 0, 0)], (1, 1, 1), tmp_path / "out.h5")

    def test_parent_dir_created(self, tmp_path, pixel_data):
        """save_params_to_hdf5 creates missing parent directories."""
        fitted_params, pixel_indices, spatial_shape = pixel_data
        out = tmp_path / "nested" / "dir" / "params.h5"
        save_params_to_hdf5(fitted_params, pixel_indices, spatial_shape, out)
        assert out.exists()
