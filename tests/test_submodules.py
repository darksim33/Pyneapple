"""Tests for submodule imports and initialization.

This module tests that required submodules can be imported and initialized correctly.
"""
from radimgarray import RadImgArray


def test_radimgarray_init():
    """Test that the RadImgArray class can be initialized with empty array."""
    # Test that the RadImgArray class can be initialized
    array = RadImgArray([])
    assert array is not None
