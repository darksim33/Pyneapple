"""Tests for ResultDict container class.

This module tests the ResultDict class which provides:

- Dictionary interface: get/set/delete operations with coordinate keys
- Array conversion: Converting stored values to numpy arrays
- Coordinate handling: Using tuples as spatial coordinate keys
- Data access: Efficient retrieval of fitted parameters
- Integration: Storage backend for fitting results

ResultDict is a specialized container for storing spatially-indexed
fitting results, mapping image coordinates to fitted parameter arrays.
"""
import numpy as np

from pyneapple.results.result_dict import ResultDict
from radimgarray import RadImgArray


class TestResultDict:
    """Test suite for ResultDict basic operations and conversions."""

    def test_get(self):
        """Test that dictionary get operations work correctly with coordinate tuple keys."""
        result_dict = ResultDict()
        result_dict[(1, 1, 1)] = 1.1

        assert result_dict[(1, 1, 1)] == 1.1
        assert result_dict.get((1, 1, 1), 0) == 1.1

    def test_get_seg(self):
        """Test that segmentation-wise storage and retrieval works with both coordinate and segment keys."""
        result_dict = ResultDict()
        result_dict[(1, 1, 1)] = 1.1
        result_dict.set_segmentation_wise({(1, 1, 1): 1})
        result_dict[1] = 1.1

        assert result_dict[(1, 1, 1)] == 1.1
        assert result_dict.get((1, 1, 1), 0) == 1.1

    def test_validate_key(self):
        """Test that keys are properly validated and converted to correct types."""
        result_dict = ResultDict()
        # result_dict[(1, 1, 1)] = 1.1
        # result_dict.set_segmentation_wise({(1, 1, 1): 1})
        result_dict[np.int32(1)] = 1.1
        for key in result_dict:
            assert isinstance(key, int)

    def test_update(self):
        """Test that update method correctly merges multiple dictionaries with coordinate keys."""
        values_1 = {(1, 1, 1): 1.5}
        values_2 = {(0, 0, 0): 1.7}
        result_dict = ResultDict()
        result_dict.update(values_1)
        assert result_dict[(1, 1, 1)] == 1.5
        result_dict.update(values_2)
        assert result_dict[(0, 0, 0)] == 1.7

    def test_update_with_seg(self):
        """Test that update method works correctly with segment number keys."""
        value_1 = {1: 1.5}
        value_2 = {2: 1.7}
        result_dict = ResultDict()
        result_dict.update(value_1)
        assert result_dict[1] == 1.5
        result_dict.update(value_2)
        assert result_dict[2] == 1.7

    def test_as_array(self):
        """Test that as_array converts ResultDict to numpy array with correct shape and values."""
        result_dict = ResultDict()
        test_array = np.array([[1.3, 2.2], [3.1, 4.4]])
        result_dict.update(
            {
                (0, 0): test_array[0, 0],
                (0, 1): test_array[0, 1],
                (1, 0): test_array[1, 0],
                (1, 1): test_array[1, 1],
            }
        )
        array = result_dict.as_array((2, 2, 1, 1))
        assert np.allclose(array, test_array[..., np.newaxis, np.newaxis])

    def test_as_rad_img_array(self):
        """Test that as_RadImgArray converts ResultDict to RadImgArray with correct shape and values."""
        result_dict = ResultDict()
        test_array = RadImgArray(np.array([[1.3, 2.2], [3.1, 4.4]]))
        result_dict.update(
            {
                (0, 0, 0, 0): test_array[0, 0],
                (0, 1, 0, 0): test_array[0, 1],
                (1, 0, 0, 0): test_array[1, 0],
                (1, 1, 0, 0): test_array[1, 1],
            }
        )
        test_array = test_array[:, :, np.newaxis, np.newaxis]
        array = result_dict.as_RadImgArray(test_array)
        assert np.allclose(array, test_array)
